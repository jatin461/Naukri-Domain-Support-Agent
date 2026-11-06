"""Independent review stage: a two-agent AutoGen RoundRobinGroupChat that runs after the CrewAI draft.

Policy_Compliance_Reviewer checks the draft against the original retrieved context; Final_Editor then
returns a structured Verdict (approved unchanged, or revised). Names use underscores because AutoGen
requires Python identifiers (the brief's hyphenated names are only examples). The team is bounded with
max_turns=2 (task message + reviewer + editor); the structured message type is registered with
custom_message_types, without which AutoGen raises "Message type ... is not registered".
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat

from src.llm.mock_llm import EDITOR, REVIEWER
from src.review.mock_autogen_client import MockAutoGenClient
from src.schemas.verdict import Verdict

REVIEWER_PROMPT = (f"You are the {REVIEWER}. Check every sentence of the DRAFT against the CONTEXT and flag claims "
                   "the context does not support.")
EDITOR_PROMPT = (f"You are the {EDITOR}. Using the reviewer's findings, either approve the draft unchanged or revise it "
                 "by removing unsupported claims, and answer with the structured verdict.")


class ReviewError(RuntimeError):
    pass


@dataclass
class ReviewResult:
    verdict: Verdict
    reviewer_text: str
    agent_messages: list = field(default_factory=list)  # [(source, message class)] after the task message
    task: str = ""

    @property
    def revised(self):
        return not self.verdict.approved


def build_task(draft, context):
    return f"### DRAFT ###\n{draft}\n### CONTEXT ###\n{context}"


def build_agents(client):
    reviewer = AssistantAgent(REVIEWER, client, system_message=REVIEWER_PROMPT)
    editor = AssistantAgent(EDITOR, client, system_message=EDITOR_PROMPT, output_content_type=Verdict)
    return reviewer, editor


def build_team(client):
    reviewer, editor = build_agents(client)
    return RoundRobinGroupChat([reviewer, editor], max_turns=2, custom_message_types=[StructuredMessage[Verdict]])


def _run_coro(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # normal case: called from a worker thread without a running loop
    with ThreadPoolExecutor(max_workers=1) as pool:  # called from inside an event loop: use a private loop
        return pool.submit(asyncio.run, coro).result()


async def review_async(draft, context, client):
    task = build_task(draft, context)
    result = await build_team(client).run(task=task)
    msgs = result.messages
    agent_msgs = [m for m in msgs if m.source in (REVIEWER, EDITOR)]
    if [m.source for m in agent_msgs] != [REVIEWER, EDITOR] or not isinstance(agent_msgs[-1], StructuredMessage):
        raise ReviewError("review team did not end with the editor's structured verdict")
    verdict = Verdict.model_validate(agent_msgs[-1].content)
    return ReviewResult(verdict, str(agent_msgs[0].content), [(m.source, type(m).__name__) for m in agent_msgs], task)


def review(draft, context, core=None):
    """Synchronous entry point used by the pipeline's worker thread."""
    try:
        return _run_coro(review_async(draft, context, MockAutoGenClient(core)))
    except ReviewError:
        raise
    except Exception as exc:
        raise ReviewError(f"review team failed: {type(exc).__name__}: {exc}") from exc
