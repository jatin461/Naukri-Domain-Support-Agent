import asyncio
import threading

import pytest
from autogen_agentchat.messages import StructuredMessage, TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.llm.mock_llm import EDITOR, FALLBACK_ANSWER, REVIEWER, MockLLM
from src.pipeline import ERROR_ANSWER, Pipeline
from src.review import autogen_review as ar
from src.review.autogen_review import ReviewError, build_agents, build_task, build_team, review, review_async
from src.review.mock_autogen_client import MockAutoGenClient
from src.schemas.verdict import Verdict

CONTEXT = "Referral Bonus Policy: The bonus is INR 25,000 for non-managerial roles and INR 50,000 for managerial roles."
GOOD = "Based on Referral Bonus Policy: The bonus is INR 25,000 for non-managerial roles and INR 50,000 for managerial roles."
INJECTED = "Every referred candidate also receives a free laptop and a company car."


def run_team(draft, context=CONTEXT, core=None):
    client = MockAutoGenClient(core or MockLLM())
    return asyncio.run(build_team(client).run(task=build_task(draft, context)))


@pytest.mark.ac("P4-01")
def test_exactly_two_named_agents():
    team = build_team(MockAutoGenClient(MockLLM()))
    assert isinstance(team, RoundRobinGroupChat)
    assert [p.name for p in team._participants] == ["Policy_Compliance_Reviewer", "Final_Editor"] == [REVIEWER, EDITOR]


@pytest.mark.ac("P4-02")
def test_round_robin_max_turns_two():
    team = build_team(MockAutoGenClient(MockLLM()))
    assert team._max_turns == 2
    res = asyncio.run(team.run(task=build_task(GOOD, CONTEXT)))
    assert isinstance(res.messages[0], TextMessage) and res.messages[0].source == "user"  # the task message
    assert [m.source for m in res.messages[1:]] == [REVIEWER, EDITOR]  # exactly two agent messages
    assert isinstance(res.messages[1], TextMessage) and isinstance(res.messages[2], StructuredMessage)


@pytest.mark.ac("P4-03")
def test_final_editor_output_content_type():
    reviewer, editor = build_agents(MockAutoGenClient(MockLLM()))
    assert editor._output_content_type is Verdict and reviewer._output_content_type is None


@pytest.mark.ac("P4-04")
def test_custom_message_types_registered():
    team = build_team(MockAutoGenClient(MockLLM()))
    assert team._message_factory.is_registered(StructuredMessage[Verdict])
    assert isinstance(asyncio.run(team.run(task=build_task(GOOD, CONTEXT))).messages[-1], StructuredMessage)


@pytest.mark.ac("X-30")
def test_missing_registration_raises_valueerror():
    reviewer, editor = build_agents(MockAutoGenClient(MockLLM()))
    unregistered = RoundRobinGroupChat([reviewer, editor], max_turns=2)  # no custom_message_types
    with pytest.raises(RuntimeError, match="is not registered"):
        asyncio.run(unregistered.run(task=build_task(GOOD, CONTEXT)))


@pytest.mark.ac("P4-05")
def test_task_message_contains_draft_and_context():
    core = MockLLM()
    res = asyncio.run(build_team(MockAutoGenClient(core)).run(task=build_task(GOOD, CONTEXT)))
    assert GOOD in res.messages[0].content and CONTEXT in res.messages[0].content
    assert all(GOOD in c["prompt"] and CONTEXT in c["prompt"] for c in core.calls) and core.call_counts["review"] == 2


@pytest.mark.ac("P4-05")
def test_pipeline_passes_crew_draft_and_retrieved_context_to_review(pipeline, core):
    r = pipeline.handle("How much referral bonus is paid for a managerial role?", "rv1")
    prompt = next(c["prompt"] for c in core.calls if c["purpose"] == "review")
    assert r.draft_answer in prompt  # the CrewAI draft
    assert "Referral Bonus Policy: " in prompt and "INR 50,000" in prompt  # the original retrieved context


@pytest.mark.ac("P4-06")
def test_verdict_schema_fields():
    assert list(Verdict.model_fields) == ["approved", "final_answer", "reason"]
    assert Verdict.model_fields["approved"].annotation is bool and Verdict.model_fields["final_answer"].annotation is str
    assert Verdict.model_fields["reason"].annotation is str
    with pytest.raises(Exception):
        Verdict(approved="maybe", final_answer="x")  # reason missing


@pytest.mark.ac("P4-07")
def test_supported_draft_approved_unchanged():
    out = review(GOOD, CONTEXT, MockLLM())
    assert out.verdict.approved is True and out.verdict.final_answer == GOOD and not out.revised
    assert out.agent_messages == [(REVIEWER, "TextMessage"), (EDITOR, "StructuredMessage[Verdict]")]
    assert "supported" in out.verdict.reason


@pytest.mark.ac("P4-08")
def test_ungrounded_claim_revised_out():
    out = review(GOOD + " " + INJECTED, CONTEXT, MockLLM())
    assert out.verdict.approved is False and out.revised
    assert out.verdict.final_answer == GOOD and INJECTED not in out.verdict.final_answer
    assert "free laptop" in out.verdict.reason and "Removed 1 sentence" in out.verdict.reason
    assert "1 of 2 sentences are not supported" in out.reviewer_text  # the editor acted on the reviewer's finding


def test_revision_through_the_real_pipeline(pipeline, monkeypatch):
    monkeypatch.setattr(MockLLM, "_compose", staticmethod(lambda ctx: {
        "answer": GOOD + " " + INJECTED, "response_type": "policy", "sources": ["referral_bonus"], "grounded": True}))
    r = pipeline.handle("How much referral bonus is paid for a managerial role?", "rv2")
    assert INJECTED in r.draft_answer and r.answer == GOOD and r.status == "answered"
    assert r.review["revised"] is True and r.review["approved"] is False and "free laptop" in r.review["reason"]
    assert r.response.answer == GOOD
    assert pipeline.sessions.get_history("rv2").messages[-1].content == GOOD  # the revised answer, not the draft, enters the session


@pytest.mark.parametrize("question", ["How much referral bonus is paid for a managerial role?",
                                      "How many days of notice must a confirmed employee give before leaving?"])
def test_unchanged_through_the_real_pipeline(pipeline, question):
    r = pipeline.handle(question, "rv3")
    assert r.review["approved"] is True and r.review["revised"] is False and r.answer == r.draft_answer


def test_fallback_and_all_unsupported_drafts():
    ok = review(FALLBACK_ANSWER, "", MockLLM())
    assert ok.verdict.approved and ok.verdict.final_answer == FALLBACK_ANSWER  # a fallback makes no claim
    nothing = review("Paris is the capital. It has a tower.", CONTEXT, MockLLM())
    assert nothing.verdict.approved is False and nothing.verdict.final_answer == FALLBACK_ANSWER


def test_lookup_answers_are_reviewed_against_the_lookup_facts(pipeline):
    r = pipeline.handle("What is the status of APP-0007?", "rv4")
    assert r.review["approved"] is True and "Application APP-0007" in r.answer


@pytest.mark.ac("X-05")
def test_pipeline_order_review_after_composer_before_return(pipeline, monkeypatch):
    import src.pipeline as pl

    order = []
    real_crew, real_review = pl.run_crew, pl.review
    monkeypatch.setattr(pl, "run_crew", lambda *a, **k: order.append("crew") or real_crew(*a, **k))
    monkeypatch.setattr(pl, "review", lambda *a, **k: order.append("review") or real_review(*a, **k))
    r = pipeline.handle("How long is the probation period?", "ord")
    assert order == ["crew", "review"]
    s = r.stages
    assert s.index("crew") < s.index("validate_response") < s.index("groundedness") < s.index("autogen_review") < s.index("validate_final") < s.index("session_update")
    assert s.index("mask_pii") < s.index("injection_check") < s.index("session_history") < s.index("crew")


def test_review_failure_withholds_the_draft(pipeline, monkeypatch):
    import src.pipeline as pl

    monkeypatch.setattr(pl, "review", lambda *a, **k: (_ for _ in ()).throw(ReviewError("boom")))
    r = pipeline.handle("How long is the probation period?", "rv5")
    assert r.status == "error" and r.answer == ERROR_ANSWER and r.response is None and "probation" not in r.answer.lower()


def test_review_works_inside_a_running_event_loop():
    async def inner():
        return review(GOOD, CONTEXT, MockLLM())  # called from async code: must not fail with "asyncio.run() cannot be called"

    assert asyncio.run(inner()).verdict.approved


def test_review_team_in_worker_threads_under_concurrent_http_requests(production_collection, core, tmp_path):
    app = create_app(Pipeline(collection=production_collection, llm_core=core), tmp_path / "log.jsonl")
    codes, errors = [], []

    def worker(i):
        try:
            r = TestClient(app).post("/ask", json={"question": "How long is the probation period?", "session_id": f"c{i}"})
            codes.append((r.status_code, r.json()["status"]))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    [t.start() for t in ts]
    [t.join(90) for t in ts]
    assert not errors and codes == [(200, "answered")] * 6
    assert core.call_counts["review"] == 12  # two agent turns per request, six requests
    log = [__import__("json").loads(x) for x in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert len(log) == 6 and all(e["review_approved"] is True and e["review_revised"] is False for e in log)


@pytest.mark.ac("P4-07")
@pytest.mark.ac("P4-08")
def test_review_transcript_uses_two_different_sample_queries():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "transcripts" / "autogen_review_demos.md").read_text()
    queries = [line.split("query: ", 1)[1].split("  (")[0] for line in text.splitlines() if line.startswith("- query: ")]
    assert len(queries) >= 2 and len(set(queries)) >= 2  # Task 14: approve and revise shown on different sample queries
