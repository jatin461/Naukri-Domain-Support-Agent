"""Deterministic request routing (decision D6) shared by the session resolver and the crew builder.

The policy keywords are the stems of the 12 knowledge-base topic names (a test checks every topic is covered).
Heuristics (documented limits): a request goes to the Lookup Agent when it contains a well-formed
record ID; the Retrieval Agent is added when the text also contains a policy keyword (so "status of
APP-0007 and the probation policy" runs all three agents) or when there is no record ID at all.
"""
import re
from dataclasses import dataclass

from src.tools.application_lookup import detect_record_id

POLICY_HINTS = re.compile(
    r"eligib|interview|offer|negotiat|counter|background|verif|notice|referral|bonus|transfer|probation|remote|"
    r"work from home|diversity|inclusi|exit|retention|retain|delet|policy|policies|resign",
    re.IGNORECASE,
)
APPLICATION_INTENT = re.compile(r"\b(status|salary|escalat\w*|priority|stage|progress|category)\b", re.IGNORECASE)
FOLLOW_UP = re.compile(r"^\s*(and|also|then|what about|how about|what if)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Route:
    record_id: str | None
    needs_lookup: bool
    needs_policy: bool


def has_policy_hint(text):
    return bool(POLICY_HINTS.search(text))


def has_application_intent(text):
    return bool(APPLICATION_INTENT.search(text)) and not has_policy_hint(text)


def route(question):
    rid = detect_record_id(question)
    return Route(rid, rid is not None, rid is None or has_policy_hint(question))
