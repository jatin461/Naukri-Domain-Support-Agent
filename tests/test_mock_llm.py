import json
import warnings
from types import SimpleNamespace

import pytest
from crewai import Agent, Crew, Task
from crewai.llms.base_llm import BaseLLM
from pydantic import BaseModel, Field

from src.config import settings
from src.llm.mock_crewai_llm import MockCrewAILLM, make_mock_crewai_llm
from src.llm.mock_llm import MockLLM, ToolSpec, choose_tool, fill_arguments, get_mock_llm
from src.rag import vector_store
from src.rag.grounded_generation import generate_grounded_answer
from src.rag.kb_loader import load_kb
from src.tools.application_lookup import CheckJobApplicationStatusTool
from src.tools.rag_tool import RagLookupTool

SYSTEM = (
    "You are Agent. goal\nYou ONLY have access to the following tools:\n\nTool Name: x\n\n"
    "Use the following format:\nThought: you should always think about what to do\nAction: the action to take\n"
    "Action Input: the input to the action\nObservation: the result of the action\n"
)
TASK = "\nCurrent Task: {t}\n\nThis is the expected criteria for your final answer: answer\nyou MUST return the actual complete content"


class _In(BaseModel):
    query: str


class _Rec(BaseModel):
    record_id: str = Field(..., pattern=r"^APP-\d{1,6}$")


def _tool(name, schema):
    return SimpleNamespace(name=name, description="d", args_schema=schema)


def _msgs(task, assistant=None):
    m = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK.format(t=task)}]
    if assistant:
        m += [{"role": "assistant", "content": assistant}, {"role": "user", "content": "Analyze the tool result."}]
    return m


def _llm():
    return make_mock_crewai_llm(MockLLM())


@pytest.mark.ac("P2-08")
def test_is_basellm_subclass():
    assert issubclass(MockCrewAILLM, BaseLLM) and BaseLLM.__module__ == "crewai.llms.base_llm"
    assert not getattr(MockCrewAILLM(model="m"), "supports_function_calling", lambda: False)()


@pytest.mark.ac("P2-09")
def test_first_turn_ignores_template_observation():
    out = _llm().call(_msgs("Question: How are interviews scheduled?"), from_agent=SimpleNamespace(tools=[_tool("rag_lookup", _In)]))
    assert "Observation: the result of the action" in SYSTEM  # the trap is really there
    assert "Action: rag_lookup" in out and "Final Answer" not in out
    assert json.loads(out.split("Action Input: ")[1]) == {"query": "How are interviews scheduled?"}


@pytest.mark.ac("P2-09")
def test_final_answer_after_real_observation():
    assistant = 'Thought: x\nAction: rag_lookup\nAction Input: {"query": "q"}\nObservation: {"grounded": true, "answer": "REAL"}'
    out = _llm().call(_msgs("Question: q", assistant), from_agent=SimpleNamespace(tools=[_tool("rag_lookup", _In)]))
    assert out.startswith("Thought: I now know the final answer\nFinal Answer: ") and "REAL" in out and "Action:" not in out


@pytest.mark.ac("P2-10")
def test_rag_lookup_named_tool_not_misrouted():
    agent = SimpleNamespace(tools=[_tool("rag_lookup", _In)])
    out = _llm().call(_msgs("Question: What is the status process for APP-0007 policies?"), from_agent=agent)
    assert json.loads(out.split("Action Input: ")[1]).keys() == {"query"}  # name says lookup, schema says query


@pytest.mark.ac("P2-10")
def test_args_follow_declared_schema():
    # names are swapped on purpose: the policy-sounding tool takes a record_id, the id-sounding tool takes a query
    rec_tool = _tool("search_policy", _Rec)
    q_tool = _tool("check_job_application_status", _In)
    out = _llm().call(_msgs("Status of application app-0007"), from_agent=SimpleNamespace(tools=[q_tool, rec_tool]))
    assert "Action: search_policy" in out and json.loads(out.split("Action Input: ")[1]) == {"record_id": "APP-0007"}
    out = _llm().call(_msgs("Question: how long is probation?"), from_agent=SimpleNamespace(tools=[rec_tool, q_tool]))
    assert "Action: check_job_application_status" in out
    assert json.loads(out.split("Action Input: ")[1]) == {"query": "how long is probation?"}
    spec = ToolSpec("t", "d", _Rec.model_json_schema()["properties"])
    assert fill_arguments(spec, "no id in this text") == {"record_id": ""}
    assert choose_tool([ToolSpec("a", "", {"query": {"type": "string"}}), spec], "see APP-0001").name == "t"


def test_mock_records_prompt_and_counts():
    core = MockLLM()
    core.complete("crew", _msgs("Question: q"), [ToolSpec("rag_lookup", "d", {"query": {"type": "string"}})])
    assert core.call_counts["crew"] == 1 and "Current Task: Question: q" in core.calls[-1]["prompt"]
    with pytest.raises(ValueError):
        core.complete("unknown_purpose", "x")


def test_mock_flag_is_the_gate():
    assert settings.MOCK_LLM is True


# ---------- real CrewAI kickoff with the real tools ----------


class _Resp(BaseModel):
    answer: str
    response_type: str
    sources: list[str] = []
    grounded: bool = False
    record_id: str | None = None
    status: str | None = None
    expected_salary_inr: int | None = None
    escalation_score: float | None = None
    escalation_recommended: bool | None = None


@pytest.fixture(scope="module")
def collection(tmp_path_factory):
    client = vector_store.get_client(tmp_path_factory.mktemp("chroma"))
    return vector_store.build_collections(client, load_kb())[settings.RAG_STRATEGY][0]


def _crew(collection, tasks_for):
    core = MockLLM()
    llm = make_mock_crewai_llm(core)
    thr = settings.get_similarity_threshold()
    rag = RagLookupTool(search_fn=lambda q: generate_grounded_answer(q, collection, thr, llm=core))
    lookup = CheckJobApplicationStatusTool()
    mk = lambda role, tools: Agent(role=role, goal=role, backstory=role, tools=tools, llm=llm, allow_delegation=False)
    r, l, c = mk("Retrieval Agent", [rag]), mk("Lookup Agent", [lookup]), mk("Response Composer", [])
    tasks = tasks_for(r, l, c)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = Crew(agents=[r, l, c], tasks=tasks, memory=False, verbose=False).kickoff()
    return out, rag, lookup, core


def _tasks_policy(r, l, c):
    t1 = Task(description="Find policy evidence.\nQuestion: How much referral bonus is paid for a managerial role?", expected_output="evidence", agent=r)
    return [t1, Task(description="Compose the final answer.", expected_output="json", agent=c, context=[t1], output_pydantic=_Resp)]


def _tasks_lookup(rid):
    def f(r, l, c):
        t1 = Task(description=f"Look up the application.\nQuestion: What is the status of application {rid}?", expected_output="status", agent=l)
        return [t1, Task(description="Compose the final answer.", expected_output="json", agent=c, context=[t1], output_pydantic=_Resp)]
    return f


@pytest.mark.ac("P2-06")
def test_rag_tool_invoked_via_kickoff(collection):
    out, rag, lookup, core = _crew(collection, _tasks_policy)
    assert [(i["tool"], i["args"]) for i in rag.invocations] == [("rag_lookup", {"query": "How much referral bonus is paid for a managerial role?"})]
    assert rag.invocations[0]["result"]["sources"] == ["referral_bonus"]
    assert lookup.invocations == []
    assert isinstance(out.pydantic, _Resp) and out.pydantic.grounded and out.pydantic.sources == ["referral_bonus"]
    assert "INR 50,000" in out.pydantic.answer and out.pydantic.response_type == "policy"
    assert core.call_counts["grounded_answer"] == 1


@pytest.mark.ac("P2-07")
def test_lookup_tool_invoked_via_kickoff(collection):
    out, rag, lookup, _ = _crew(collection, _tasks_lookup("APP-0007"))
    assert [(i["tool"], i["args"]) for i in lookup.invocations] == [("check_job_application_status", {"record_id": "APP-0007"})]
    assert lookup.invocations[0]["result"]["found"] is True
    assert rag.invocations == []
    p = out.pydantic
    assert p.record_id == "APP-0007" and p.response_type == "lookup" and p.status and p.expected_salary_inr
    assert 0 <= p.escalation_score <= 1 and "Escalation score" in p.answer


def test_unknown_id_via_kickoff(collection):
    out, _, lookup, _ = _crew(collection, _tasks_lookup("APP-9999"))
    assert lookup.invocations[0]["args"] == {"record_id": "APP-9999"}
    assert out.pydantic.status is None and "No application found" in out.pydantic.answer


def test_composer_combines_policy_and_lookup(collection):
    def both(r, l, c):
        t1 = Task(description="Evidence.\nQuestion: How long is the probation period?", expected_output="e", agent=r)
        t2 = Task(description="Lookup.\nQuestion: status of APP-0003", expected_output="s", agent=l)
        return [t1, t2, Task(description="Compose.", expected_output="json", agent=c, context=[t1, t2], output_pydantic=_Resp)]
    out, rag, lookup, _ = _crew(collection, both)
    assert rag.invocations and lookup.invocations
    assert out.pydantic.response_type == "policy_and_lookup" and out.pydantic.record_id == "APP-0003" and out.pydantic.sources


def test_out_of_scope_policy_falls_back_through_crew(collection):
    def oos(r, l, c):
        t1 = Task(description="Evidence.\nQuestion: What is the capital of France?", expected_output="e", agent=r)
        return [t1, Task(description="Compose.", expected_output="json", agent=c, context=[t1], output_pydantic=_Resp)]
    out, _, _, core = _crew(collection, oos)
    assert out.pydantic.grounded is False and out.pydantic.answer.startswith("I don't know")
    assert core.call_counts["grounded_answer"] == 0  # fallback needs no generation call


def test_shared_core_is_one_instance():
    assert get_mock_llm() is get_mock_llm()


def test_overlapping_windows_are_merged_into_continuous_text():
    from src.llm.mock_llm import merge_overlapping

    a, b, c = "one two three four five six", "four five six seven eight nine", "seven eight nine ten eleven twelve"
    assert merge_overlapping([c, a, b]) == ["one two three four five six seven eight nine ten eleven twelve"]
    assert merge_overlapping(["alpha beta gamma delta.", "unrelated sentence here."]) == ["alpha beta gamma delta.", "unrelated sentence here."]
    assert merge_overlapping(["a b c", "x b c"]) == ["a b c", "x b c"]  # an overlap shorter than 3 words is not a join
    assert merge_overlapping(["same text here", "same text here"]) == ["same text here"]
