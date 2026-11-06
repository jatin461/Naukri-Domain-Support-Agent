"""The single request path shared by HTTP and WebSocket (architecture §9.1).

Stages, in order: validate, trace_session, mask_pii, injection_check, budget_check, session_history,
[resolve_reference, crew, validate_response, groundedness, autogen_review, validate_final], session_update.
The request log is written by the API layer. The grounded-generation cache lives inside the RAG tool's search function,
so it never bypasses any stage above or below it.
Every crew draft passes the AutoGen review team before delivery; if the review fails the draft is withheld.
A rejected request (invalid or injection) never reaches the model or the session history.
"""
import time
import uuid
import warnings
from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel, Field

from src.cache.grounded_cache import GroundedCache
from src.config import settings
from src.crew.crew import CrewOutputError, run_crew
from src.crew.routing import route as route_question
from src.governance.budget import check_budget
from src.guardrails.input_guardrails import InvalidRequest, detect_bulk_request, detect_injection, mask_phone, phone_count, validate_question
from src.guardrails.output_guardrails import build_evidence, check_groundedness
from src.llm.mock_crewai_llm import make_mock_crewai_llm
from src.llm.mock_llm import FALLBACK_ANSWER, get_mock_llm
from src.memory.session_memory import CLARIFY_RECORD, SessionStore, new_session_id, resolve_reference
from src.rag.grounded_generation import generate_grounded_answer
from src.review.autogen_review import ReviewError, review
from src.schemas.crew_response import CrewResponse

ERROR_ANSWER = "Sorry, I could not produce a validated answer for this request."
INJECTION_ANSWER = "This request was blocked because it looks like an attempt to override the assistant's instructions."
BULK_ANSWER = "This assistant looks up one application at a time; it cannot list salaries or records for all applicants."


class PipelineResult(BaseModel):
    trace_id: str
    session_id: str
    status: Literal["answered", "fallback", "clarification", "rejected", "error"]
    answer: str
    draft_answer: str | None = None  # the crew's draft before any guardrail replacement
    response: CrewResponse | None = None
    rejection_reason: str | None = None
    masked_question: str | None = None
    phone_numbers_masked: int = 0
    injection_matched: list[str] = Field(default_factory=list)
    resolution: str | None = None
    groundedness: dict | None = None
    budget: dict | None = None
    cache_events: list[str] = Field(default_factory=list)
    review: dict | None = None
    tool_invocations: list[dict] = Field(default_factory=list)
    stages: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0


def load_default_collection():
    from src.rag.vector_store import get_client, get_collection

    col = get_collection(get_client(), settings.RAG_STRATEGY)
    if col.count() == 0:
        raise RuntimeError("The vector index is empty. Run: .venv/bin/python -m scripts.build_index")
    return col


class Pipeline:
    def __init__(self, collection=None, threshold=None, llm_core=None, sessions=None, token_limit=None):
        self.collection = collection if collection is not None else load_default_collection()
        self.threshold = threshold if threshold is not None else settings.get_similarity_threshold()
        self.llm_core = llm_core or get_mock_llm()
        self.sessions = sessions or SessionStore()
        self.crew_llm = make_mock_crewai_llm(self.llm_core)
        self.index_version = 0  # bumped when documents are added; part of the cache key
        self.cache = GroundedCache()
        self.token_limit = token_limit
        # LangChain's deprecation warning for RunnableWithMessageHistory is expected and deliberately not silenced.
        self._chain = RunnableWithMessageHistory(
            RunnableLambda(self._run), self.sessions.get_history,
            input_messages_key="input", history_messages_key="history",
        )

    # ---- inner stage, executed with the session history supplied by RunnableWithMessageHistory ----
    def _run(self, inp):
        ctx, history, masked = inp["ctx"], inp["history"], inp["input"]
        ctx["stages"].append("session_history")
        res = resolve_reference(masked, history)
        ctx["stages"].append("resolve_reference")
        ctx["resolution"] = res.action
        if res.action == "clarify":
            resp = CrewResponse(answer=CLARIFY_RECORD, response_type="clarification")
            ctx.update(status="clarification", response=resp, answer=resp.answer)
            return AIMessage(content=resp.answer)

        rt = route_question(res.question)
        rag_results = []

        def search(query):
            r, event = self.cache.get_or_compute(
                query, self.index_version,
                lambda: generate_grounded_answer(query, self.collection, self.threshold, llm=self.llm_core),
            )
            rag_results.append(r)
            ctx.setdefault("cache_events", []).append(event)
            return r

        ctx["stages"].append("crew")
        try:
            run = run_crew(res.question, rt, self.crew_llm, search)
        except CrewOutputError as exc:
            ctx["stages"].append("validate_response")
            ctx.update(status="error", answer=ERROR_ANSWER, error=str(exc))
            return AIMessage(content=ERROR_ANSWER)
        ctx["stages"].append("validate_response")
        ctx["draft"] = run.response.answer
        ctx["tool_invocations"] = [{"tool": i["tool"], "args": i["args"]} for i in run.rag_invocations + run.lookup_invocations]

        lookups = [i["result"] for i in run.lookup_invocations]
        verdict = check_groundedness(run.response.answer, rag_results, lookups)
        ctx["stages"].append("groundedness")
        ctx["groundedness"] = {"action": verdict.action, "reason": verdict.reason,
                               "unsupported_sentences": verdict.unsupported_sentences, "sentence_scores": verdict.sentence_scores}
        resp = run.response
        if verdict.action == "refuse":
            resp = CrewResponse(answer=FALLBACK_ANSWER, response_type="no_evidence")
        ctx["stages"].append("autogen_review")
        try:
            rv = review(resp.answer, "\n".join(build_evidence(rag_results, lookups)), self.llm_core)
        except ReviewError as exc:  # an unreviewed draft is never delivered
            ctx.update(status="error", answer=ERROR_ANSWER, error=str(exc), response=None)
            return AIMessage(content=ERROR_ANSWER)
        ctx["review"] = {"approved": rv.verdict.approved, "revised": rv.revised, "reason": rv.verdict.reason,
                         "agent_messages": [list(m) for m in rv.agent_messages]}
        ctx["stages"].append("validate_final")
        if rv.verdict.final_answer.strip() == FALLBACK_ANSWER:
            resp = CrewResponse(answer=FALLBACK_ANSWER, response_type="no_evidence")
        else:
            resp = CrewResponse.model_validate({**resp.model_dump(), "answer": rv.verdict.final_answer})
        fallback = resp.response_type == "no_evidence" or (resp.response_type == "policy" and not resp.grounded)
        ctx.update(status="fallback" if fallback else "answered", response=resp, answer=resp.answer)
        record_id = resp.record_id if resp.status else res.carried  # only records that actually exist are remembered
        return AIMessage(
            content=resp.answer,
            additional_kwargs={"record_id": record_id, "policy_question": res.question if rt.needs_policy and not fallback else None},
        )

    def on_index_changed(self):
        self.index_version += 1
        self.cache.clear()

    # ---- public entry point ----
    def handle(self, question, session_id=None, trace_id=None):
        t0 = time.perf_counter()
        trace_id = trace_id or uuid.uuid4().hex
        session_id = session_id or new_session_id()
        stages = ["validate"]

        def finish(**kw):
            return PipelineResult(trace_id=trace_id, session_id=session_id, stages=stages,
                                  duration_ms=round((time.perf_counter() - t0) * 1000, 3), **kw)

        try:
            q = validate_question(question)
        except InvalidRequest as exc:
            return finish(status="rejected", answer=str(exc), rejection_reason="invalid_request")
        stages.append("trace_session")
        masked = mask_phone(q)
        n_phone = phone_count(q)
        stages.append("mask_pii")
        inj = detect_injection(masked)
        stages.append("injection_check")
        if inj.blocked:
            return finish(status="rejected", answer=INJECTION_ANSWER, rejection_reason="prompt_injection",
                          masked_question=masked, phone_numbers_masked=n_phone, injection_matched=inj.matched)
        if detect_bulk_request(masked):  # the lookup tool reads one record at a time; enumeration requests are unsupported
            return finish(status="rejected", answer=BULK_ANSWER, rejection_reason="unsupported_bulk_request",
                          masked_question=masked, phone_numbers_masked=n_phone)

        decision = check_budget(masked, self.token_limit)
        stages.append("budget_check")
        if not decision.allowed:  # rejected before the crew is built and before any model call
            return finish(status="rejected", answer=f"Request rejected: estimated {decision.estimated_tokens} tokens exceeds the per-request limit of {decision.limit}.",
                          rejection_reason="budget_exceeded", masked_question=masked, phone_numbers_masked=n_phone, budget=decision.as_dict())

        ctx = {"stages": stages}
        self._chain.invoke({"input": masked, "ctx": ctx}, config={"configurable": {"session_id": session_id}})
        stages.append("session_update")
        return finish(status=ctx["status"], answer=ctx["answer"], draft_answer=ctx.get("draft"), response=ctx.get("response"), masked_question=masked,
                      phone_numbers_masked=n_phone, resolution=ctx.get("resolution"), groundedness=ctx.get("groundedness"), review=ctx.get("review"), budget=decision.as_dict(), cache_events=ctx.get("cache_events", []),
                      tool_invocations=ctx.get("tool_invocations", []),
                      rejection_reason=ctx.get("error"))
