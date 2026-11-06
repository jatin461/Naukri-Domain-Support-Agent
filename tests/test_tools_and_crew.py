import pytest
from crewai import Crew
from pydantic import ValidationError

from src.config import settings
from src.crew.agents import build_agents
from src.crew.crew import CrewOutputError, run_crew
from src.crew.routing import Route, route
from src.crew.tasks import build_tasks
from src.llm.mock_crewai_llm import make_mock_crewai_llm
from src.llm.mock_llm import MockLLM
from src.pipeline import ERROR_ANSWER
from src.rag.grounded_generation import generate_grounded_answer
from src.schemas.crew_response import CrewResponse
from src.tools.application_lookup import CheckJobApplicationStatusTool
from src.tools.rag_tool import RagLookupTool


def _stub_agents():
    llm = make_mock_crewai_llm(MockLLM())
    return build_agents(llm, RagLookupTool(search_fn=lambda q: None), CheckJobApplicationStatusTool())


def _runner(production_collection):
    core = MockLLM()
    llm = make_mock_crewai_llm(core)
    search = lambda q: generate_grounded_answer(q, production_collection, settings.get_similarity_threshold(), llm=core)
    return lambda question: run_crew(question, route(question), llm, search), core


@pytest.mark.ac("P2-04")
def test_three_distinct_agents():
    a = _stub_agents()
    roles = [x.role for x in a]
    assert roles == ["Retrieval Agent", "Lookup Agent", "Response Composer"] and len(set(roles)) == 3
    assert all(x.allow_delegation is False for x in a)
    assert len({id(x) for x in a}) == 3


@pytest.mark.ac("P2-05")
def test_kickoff_returns_output(production_collection, monkeypatch):
    calls = []
    original = Crew.kickoff
    monkeypatch.setattr(Crew, "kickoff", lambda self, *a, **k: calls.append(1) or original(self, *a, **k))
    run, _ = _runner(production_collection)
    out = run("How long is the probation period?")
    assert calls == [1] and out.raw and isinstance(out.response, CrewResponse)


@pytest.mark.ac("P2-06")
@pytest.mark.ac("X-08")
def test_rag_tool_invoked_on_policy_query(production_collection):
    run, core = _runner(production_collection)
    out = run("How long is the probation period?")
    assert [i["tool"] for i in out.rag_invocations] == ["rag_lookup"] and out.lookup_invocations == []
    assert out.rag_invocations[0]["args"] == {"query": "How long is the probation period?"}
    assert out.response.sources and out.response.response_type == "policy"


@pytest.mark.ac("P2-07")
@pytest.mark.ac("X-08")
def test_lookup_tool_invoked_on_status_query(production_collection):
    run, core = _runner(production_collection)
    out = run("What is the status of APP-0012?")
    assert [i["tool"] for i in out.lookup_invocations] == ["check_job_application_status"] and out.rag_invocations == []
    assert out.lookup_invocations[0]["args"] == {"record_id": "APP-0012"}
    assert out.response.record_id == "APP-0012" and core.call_counts["grounded_answer"] == 0


def test_routing_builds_per_request_task_lists():
    a = _stub_agents()
    cases = {
        "How long is probation?": ["Retrieval Agent", "Response Composer"],
        "What is the status of APP-0007?": ["Lookup Agent", "Response Composer"],
        "What is the status of APP-0007 and the probation policy?": ["Retrieval Agent", "Lookup Agent", "Response Composer"],
        "What is the capital of France?": ["Retrieval Agent", "Response Composer"],
    }
    for q, roles in cases.items():
        assert [t.agent.role for t in build_tasks(a, q, route(q))] == roles
        assert build_tasks(a, q, route(q))[0].description.endswith(f"Question: {q}")
    assert route("status of app-0007") == Route("APP-0007", True, False)


@pytest.mark.ac("P2-14")
@pytest.mark.ac("X-28")
def test_every_crew_response_validated(pipeline):
    tasks = build_tasks(_stub_agents(), "q", route("q"))
    from src.schemas.crew_response import response_format
    assert response_format is CrewResponse and tasks[-1].output_pydantic is response_format  # the declared response_format
    for i, q in enumerate(["How long is probation?", "status of APP-0007", "status of APP-9999",
                           "status of APP-0003 and the probation policy", "What is the capital of France?", "and its salary?"]):
        r = pipeline.handle(q, f"v{i}")
        assert r.response is not None
        assert CrewResponse.model_validate(r.response.model_dump()) == r.response


@pytest.mark.ac("P2-14")
def test_malformed_output_fails_safely(pipeline, monkeypatch):
    # a Composer payload that violates the schema (status without the rest of the record)
    monkeypatch.setattr(MockLLM, "_compose", staticmethod(lambda ctx: {"answer": "x", "response_type": "lookup", "record_id": "APP-1", "status": "Applied"}))
    r = pipeline.handle("status of APP-0007", "bad1")
    assert r.status == "error" and r.answer == ERROR_ANSWER and r.response is None
    assert "failed schema validation" in r.rejection_reason or "crew execution failed" in r.rejection_reason
    monkeypatch.setattr(MockLLM, "_compose", staticmethod(lambda ctx: {"answer": "", "response_type": "policy"}))
    assert pipeline.handle("How long is probation?", "bad2").status == "error"


def test_run_crew_raises_crew_output_error(production_collection, monkeypatch):
    monkeypatch.setattr(MockLLM, "_compose", staticmethod(lambda ctx: {"answer": "x", "response_type": "nonsense"}))
    run, _ = _runner(production_collection)
    with pytest.raises(CrewOutputError):
        run("How long is probation?")


def test_crew_response_schema_rules():
    ok = {"answer": "a", "response_type": "policy"}
    assert CrewResponse(**ok).sources == []
    for bad in [{**ok, "answer": ""}, {**ok, "response_type": "weird"}, {**ok, "escalation_score": 1.5},
                {**ok, "expected_salary_inr": -1}, {**ok, "status": "Applied"}, {**ok, "response_type": "lookup"}]:
        with pytest.raises(ValidationError):
            CrewResponse(**bad)
    full = {"answer": "a", "response_type": "lookup", "record_id": "APP-0001", "status": "Applied",
            "expected_salary_inr": 1, "escalation_score": 0.5, "escalation_recommended": True}
    assert CrewResponse(**full).record_id == "APP-0001"
