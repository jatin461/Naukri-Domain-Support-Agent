"""Builds and runs one crew per request and validates its response against CrewResponse."""
import warnings
from dataclasses import dataclass, field

from crewai import Crew
from pydantic import ValidationError

from src.crew.agents import build_agents
from src.crew.tasks import build_tasks
from src.schemas.crew_response import CrewResponse, response_format
from src.tools.application_lookup import CheckJobApplicationStatusTool
from src.tools.rag_tool import RagLookupTool


class CrewOutputError(RuntimeError):
    """The crew failed or returned something that does not conform to CrewResponse."""


@dataclass
class CrewRun:
    response: CrewResponse
    raw: str
    rag_invocations: list = field(default_factory=list)
    lookup_invocations: list = field(default_factory=list)
    tool_names_by_agent: dict = field(default_factory=dict)


def run_crew(question, route, llm, search_fn):
    rag_tool = RagLookupTool(search_fn=search_fn)
    lookup_tool = CheckJobApplicationStatusTool()
    agents = build_agents(llm, rag_tool, lookup_tool)
    tasks = build_tasks(agents, question, route)
    crew = Crew(agents=list(agents), tasks=tasks, memory=False, verbose=False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # CrewAI-internal deprecations only, scoped to the kickoff call
            out = crew.kickoff()
        candidate = out.pydantic.model_dump() if out.pydantic is not None else None
        response = response_format.model_validate(candidate) if candidate is not None else response_format.model_validate_json(out.raw)
    except (ValidationError, ValueError) as exc:
        raise CrewOutputError(f"crew response failed schema validation: {type(exc).__name__}") from exc
    except Exception as exc:
        raise CrewOutputError(f"crew execution failed: {type(exc).__name__}: {exc}") from exc
    return CrewRun(
        response, out.raw, rag_tool.invocations, lookup_tool.invocations,
        {a.role: [t.name for t in a.tools] for a in agents},
    )
