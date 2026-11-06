"""In-process session memory (LangChain InMemoryChatMessageHistory) and the reference resolver (D7).

History holds only masked user text and the assistant answers. The assistant message carries
`additional_kwargs` {"record_id", "policy_question"}, which the resolver reads on the next turn:
  * an application question without a record ID reuses the session's most recent record ID;
    with none in the session it gets a controlled "which application?" clarification;
  * a bare follow-up ("and during probation?") is attached to the session's last policy question.
Limits: the resolver assumes the most recent record/topic in the session is meant; it does not
disambiguate between several earlier records and only understands the cue words in routing.py.
"""
import threading
import uuid
from dataclasses import dataclass

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage

from src.crew.routing import FOLLOW_UP, has_application_intent, has_policy_hint
from src.tools.application_lookup import detect_record_id

CLARIFY_RECORD = "Which application do you mean? Please include a record ID such as APP-0007."


def new_session_id():
    return f"sess-{uuid.uuid4().hex[:12]}"


class SessionStore:
    """Histories keyed by session ID. A lock guards the dict; each history is only used by its own session."""

    def __init__(self):
        self._histories = {}
        self._lock = threading.Lock()

    def get_history(self, session_id):
        with self._lock:
            return self._histories.setdefault(session_id, InMemoryChatMessageHistory())

    def exists(self, session_id):
        with self._lock:
            return session_id in self._histories

    def drop(self, session_id):
        with self._lock:
            self._histories.pop(session_id, None)

    def __len__(self):
        return len(self._histories)


@dataclass(frozen=True)
class Resolution:
    question: str
    action: str  # "unchanged" | "resolved_record" | "resolved_follow_up" | "clarify"
    carried: str | None = None


def _last_kwarg(history, key):
    for m in reversed(history):
        if isinstance(m, AIMessage) and m.additional_kwargs.get(key):
            return m.additional_kwargs[key]
    return None


def resolve_reference(question, history):
    if detect_record_id(question):
        return Resolution(question, "unchanged")
    if has_application_intent(question):
        rid = _last_kwarg(history, "record_id")
        if rid is None:
            return Resolution(question, "clarify")
        return Resolution(f"{question} (regarding application {rid})", "resolved_record", rid)
    if FOLLOW_UP.match(question) and not has_policy_hint(question):
        prev = _last_kwarg(history, "policy_question")
        if prev:
            return Resolution(f"{prev} Follow-up: {question}", "resolved_follow_up", prev)
    return Resolution(question, "unchanged")
