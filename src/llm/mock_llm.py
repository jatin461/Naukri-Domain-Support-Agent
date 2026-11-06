"""Shared deterministic mock LLM core (MOCK_LLM mode; no network, no keys).

One implementation serves every LLM consumer; framework adapters stay thin:
  purpose "grounded_answer"  grounded generation (extractive answer from the prompt's context chunks)
  purpose "crew"             CrewAI ReAct steps (src/llm/mock_crewai_llm.py adapts BaseLLM to this)
  purpose "review"           AutoGen review team (src/review/mock_autogen_client.py adapts ChatCompletionClient to this)
  purpose "judge"            evaluation judge (registered by evaluation/judge.py)

`complete(purpose, messages, tools, response_model)` takes chat messages like a real LLM and
returns text. Every call is recorded (`calls`, `call_counts`) so tests and the cache demo can
show what the model received and how often it ran.

ReAct rules for "crew" (CrewAI 1.15.x, plain BaseLLM => text ReAct path):
  * `Observation:` is read only from the LAST assistant message. The system prompt contains the
    literal template text "Observation: the result of the action", which must never be parsed.
  * Tool arguments are built from each tool's declared JSON-schema properties: a property that
    declares a `pattern` is filled with the first match of that pattern in the task text; any
    other string property receives the question text. Tool names are never inspected.
  * A tool-using agent calls its tool once, then returns the observation as its final answer.
  * An agent without tools (the Composer) reads the upstream context block and writes the answer.
Composer payload keys: answer, response_type, sources, grounded, record_id, status,
expected_salary_inr, escalation_score, escalation_recommended.
"""
import json
import re
import threading
from collections import Counter, deque
from dataclasses import dataclass, field

from src.config import settings
from src.guardrails.support import content_tokens, sentence_support
from src.rag.kb_loader import split_sentences

FALLBACK_ANSWER = "I don't know: the knowledge base does not contain information that answers this question."

_CHUNK_LINE = re.compile(r"^\[CHUNK \d+ \| (.*?)\] (.*)$", re.MULTILINE)
_TASK = re.compile(r"Current Task:\s*(.*?)\n\nThis is the expected", re.DOTALL)
_CONTEXT = re.compile(r"This is the context you're working with:\n(.*?)\n\nProvide your complete response:", re.DOTALL)
_OBSERVATION = re.compile(r"Action Input:.*?\nObservation:\s*(.*)\Z", re.DOTALL)
_QUESTION = re.compile(r"Question:\s*(.*)\Z", re.DOTALL)
_REVIEW_TASK = re.compile(r"### DRAFT ###\n(.*?)\n### CONTEXT ###\n(.*)\Z", re.DOTALL)
_UNSUPPORTED = re.compile(r"^UNSUPPORTED_JSON: (.*)$", re.MULTILINE)
REVIEWER, EDITOR = "Policy_Compliance_Reviewer", "Final_Editor"


@dataclass(frozen=True)
class ToolSpec:
    """Framework-neutral view of a tool: its name and the JSON-schema `properties` of its arguments."""

    name: str
    description: str
    properties: dict = field(default_factory=dict)


def _text(content):
    return content if isinstance(content, str) else json.dumps(content)


def _json_objects(text):
    dec, i, out = json.JSONDecoder(), 0, []
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        out.append(obj)
        i = end
    return out


MIN_MERGE_OVERLAP_WORDS = 3


def merge_overlapping(texts):
    """Join fixed-size windows that overlap (a suffix of one equals a prefix of another) into continuous text."""
    parts = [x.split() for x in dict.fromkeys(texts)]
    merged = True
    while merged:
        merged = False
        best = None
        for i, a in enumerate(parts):
            for j, b in enumerate(parts):
                if i == j:
                    continue
                for k in range(min(len(a), len(b)), MIN_MERGE_OVERLAP_WORDS - 1, -1):
                    if a[-k:] == b[:k]:
                        if best is None or k > best[0]:
                            best = (k, i, j)
                        break
        if best:
            k, i, j = best
            parts[i] = parts[i] + parts[j][k:]
            del parts[j]
            merged = True
    return [" ".join(p) for p in parts]


def question_text(task_text):
    m = _QUESTION.search(task_text)
    return (m.group(1) if m else task_text).strip()


def fill_arguments(tool, task_text):
    """Build tool arguments from the declared schema (never from the tool's name)."""
    args = {}
    for pname, prop in tool.properties.items():
        if prop.get("type") != "string":
            continue
        pattern = prop.get("pattern")
        if pattern:
            m = re.search(pattern.strip("^$"), task_text, re.IGNORECASE)
            args[pname] = m.group(0).upper() if m else ""
        else:
            args[pname] = question_text(task_text)
    return args


def choose_tool(tools, task_text):
    """Prefer a tool whose pattern-constrained arguments match the task text, then a tool with
    only free-text arguments, then the first tool."""
    def patterns(tool):
        return [p["pattern"] for p in tool.properties.values() if p.get("pattern")]

    def matches(tool):
        return bool(patterns(tool)) and all(re.search(p.strip("^$"), task_text, re.IGNORECASE) for p in patterns(tool))

    return next((t for t in tools if matches(t)), next((t for t in tools if not patterns(t)), tools[0]))


class MockLLM:
    def __init__(self):
        self.call_counts = Counter()
        self.calls = deque(maxlen=1000)  # {"purpose", "prompt"}; the model input exactly as received
        self._lock = threading.Lock()
        self._handlers = {"grounded_answer": self._grounded_answer, "crew": self._crew_step, "review": self._review}

    def __deepcopy__(self, memo):
        # CrewAI deep-copies agent LLMs; the core must stay one shared object so counters and recorded calls are not lost.
        return self

    def register(self, purpose, handler):
        self._handlers[purpose] = handler

    def complete(self, purpose, messages, tools=(), response_model=None):
        if purpose not in self._handlers:
            raise ValueError(f"MockLLM has no handler for purpose {purpose!r}")
        msgs = [{"role": "user", "content": messages}] if isinstance(messages, str) else list(messages)
        with self._lock:
            self.call_counts[purpose] += 1
            self.calls.append({"purpose": purpose, "prompt": "\n".join(_text(m["content"]) for m in msgs)})
        return self._handlers[purpose](msgs, list(tools), response_model)

    # ---- grounded generation ----
    def _grounded_answer(self, msgs, tools, response_model):
        prompt = _text(msgs[-1]["content"])
        pairs = _CHUNK_LINE.findall(prompt)
        titles = list(dict.fromkeys(t for t, _ in pairs))
        body = " ".join(x for title in titles for x in merge_overlapping([x for t, x in pairs if t == title]))
        return f"Based on {', '.join(titles)}: {body}"

    # ---- AutoGen review team ----
    @staticmethod
    def _unsupported_sentences(draft, context):
        """Sentences of the draft with less than GROUNDING_SUPPORT_THRESHOLD of their content words in the context."""
        if draft.strip() == FALLBACK_ANSWER:
            return []  # the fallback makes no claim
        toks = content_tokens(context)
        return [s for s in split_sentences(draft) if sentence_support(s, toks) < settings.GROUNDING_SUPPORT_THRESHOLD]

    def _review(self, msgs, tools, response_model):
        system = next(_text(m["content"]) for m in msgs if m["role"] == "system")
        task = next(_text(m["content"]) for m in msgs if m["role"] == "user" and "### DRAFT ###" in _text(m["content"]))
        draft, context = _REVIEW_TASK.search(task).groups()
        if REVIEWER in system and EDITOR not in system:
            bad = self._unsupported_sentences(draft, context)
            if not bad:
                return "Policy review: every sentence of the draft is supported by the retrieved context.\nUNSUPPORTED_JSON: []"
            return f"Policy review: {len(bad)} of {len(split_sentences(draft))} sentences are not supported by the retrieved context.\nUNSUPPORTED_JSON: {json.dumps(bad)}"
        reviewer_note = next((_text(m["content"]) for m in reversed(msgs) if m.get("name") == REVIEWER), "")
        found = _UNSUPPORTED.search(reviewer_note)
        bad = json.loads(found.group(1)) if found else self._unsupported_sentences(draft, context)
        if not bad:
            return json.dumps({"approved": True, "final_answer": draft, "reason": "Every sentence of the draft is supported by the retrieved context; delivered unchanged."})
        kept = " ".join(s for s in split_sentences(draft) if s not in bad)
        return json.dumps({"approved": False, "final_answer": kept or FALLBACK_ANSWER,
                           "reason": f"Removed {len(bad)} sentence(s) not supported by the retrieved context: {json.dumps(bad)}"})

    # ---- CrewAI ReAct ----
    @staticmethod
    def _last_observation(msgs):
        for m in reversed(msgs):
            if m["role"] == "assistant":
                mt = _OBSERVATION.search(_text(m["content"]))
                return mt.group(1).strip() if mt else None
        return None

    def _crew_step(self, msgs, tools, response_model):
        first_user = next(_text(m["content"]) for m in msgs if m["role"] == "user")
        task_m = _TASK.search(first_user)
        task_text = task_m.group(1).strip() if task_m else first_user
        obs = self._last_observation(msgs)
        if tools and obs is None:
            tool = choose_tool(tools, task_text)
            args = fill_arguments(tool, task_text)
            return f"Thought: I need to use the {tool.name} tool.\nAction: {tool.name}\nAction Input: {json.dumps(args)}"
        if tools:
            return f"Thought: I now know the final answer\nFinal Answer: {obs}"
        ctx_m = _CONTEXT.search(first_user)
        payload = self._compose(ctx_m.group(1) if ctx_m else "")
        body = response_model.model_validate(payload).model_dump_json() if response_model else payload["answer"]
        return f"Thought: I now know the final answer\nFinal Answer: {body}"

    @staticmethod
    def _compose(context):
        rag = lookup = None
        for obj in _json_objects(context):
            if "grounded" in obj:
                rag = obj
            elif "found" in obj:
                lookup = obj
        parts, payload = [], {"sources": [], "grounded": False}
        if rag:
            parts.append(rag["answer"])
            payload.update(sources=rag["sources"], grounded=rag["grounded"])
        if lookup and lookup["found"]:
            thr = lookup["escalation_threshold"]
            verdict = "recommended" if lookup["escalation_recommended"] else "not recommended"
            parts.append(
                f"Application {lookup['record_id']} is at status {lookup['status']} with expected salary "
                f"INR {lookup['expected_salary_inr']:,}. Escalation score {lookup['escalation_score']:.2f} "
                f"against threshold {thr:.2f}: escalation {verdict}."
            )
            payload.update(
                record_id=lookup["record_id"], status=lookup["status"],
                expected_salary_inr=lookup["expected_salary_inr"], escalation_score=lookup["escalation_score"],
                escalation_recommended=lookup["escalation_recommended"],
            )
        elif lookup:
            parts.append(lookup["error"])
            payload.update(record_id=lookup.get("record_id"))
        kinds = [k for k, v in (("policy", rag), ("lookup", lookup)) if v]
        payload["response_type"] = "_and_".join(kinds) if kinds else "no_evidence"
        payload["answer"] = " ".join(parts) if parts else FALLBACK_ANSWER
        return payload


_shared = MockLLM()


def get_mock_llm():
    return _shared
