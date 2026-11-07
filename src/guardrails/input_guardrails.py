"""Input-side guardrails: request validation, phone masking and prompt-injection detection.

Phone masking scope: the fixed-format phone number of the contact details,
canonically `+91-NNNNN-NNNNN` in a `Contact: ...` line. Variants `+91 NNNNN NNNNN`, `+91NNNNNNNNNN`,
`NNNNN-NNNNN`, `NNNNN NNNNN` and ten contiguous digits are masked wherever they stand as their own token
(so any unrelated ten-digit number is masked too). A digit run glued to letters or other digits, such as
the inside of a hex trace ID like `af9876543210ab`, is not a phone number and is left intact. Candidate names, expected salary and background-check
results are NOT masked: out of scope for this keyless pattern-based masker.

Injection detection scope and limits: a small deterministic list of regular expressions for common
English override / prompt-exfiltration / role-reset phrasings (override verbs such as ignore, disregard,
forget, set aside; objects such as instructions, rules, guidance). It does not detect paraphrases that
avoid those phrasings, other languages, obfuscated or encoded text, or instructions hidden in
retrieved documents. Policy: a matching request is rejected (not rewritten) before any model sees it.
A second, separate check rejects bulk-data requests (list/show/export all salaries, candidates, applications).
"""
import re
from dataclasses import dataclass, field

from src.cache.grounded_cache import normalize_query
from src.config import settings

PHONE_MASK = "[PHONE_MASKED]"
_PHONE = re.compile(r"(?<![A-Za-z0-9_])(?:\+91[\s-]?)?\d{5}[\s-]?\d{5}(?![A-Za-z0-9_])")


def mask_phone(text):
    """The single masking function used for model input and for log content."""
    return _PHONE.sub(PHONE_MASK, text)


def phone_count(text):
    return len(_PHONE.findall(text))


_OVERRIDE_VERBS = (
    r"(ignore|disregard|forget|override|overrule|discard|dismiss|drop|ditch|scrap|abandon|bypass|circumvent|neglect|overlook|"
    r"cancel|nullify|void|revoke|erase|delete|wipe|clear|throw away|throw out|toss|get rid of|set aside|put aside|push aside|"
    r"pay no attention to|stop following|stop obeying|stop adhering to|no longer follow|no longer obey|do not obey|don't obey)"
)
_OVERRIDE_QUALIFIERS = r"(previous|prior|above|earlier|all|any|every|everything|whatever|your|the|you were|you have been|you've been|given|original|initial|system)"
_OVERRIDE_OBJECTS = r"(instructions?|prompts?|rules?|guidelines?|guidance|directions?|directives?|guardrails?|restrictions?|constraints?|programming|briefing)"

_INJECTION_PATTERNS = {
    "ignore_previous_instructions": rf"\b{_OVERRIDE_VERBS}\b.{{0,30}}\b{_OVERRIDE_QUALIFIERS}\b.{{0,30}}\b{_OVERRIDE_OBJECTS}\b",
    "reveal_system_prompt": r"\b(reveal|show|print|display|repeat|leak)\b.{0,30}\b(system prompt|hidden prompt|your instructions|your prompt|initial prompt)\b",
    "role_reset": r"\b(you are now|from now on you are|act as|pretend (to be|you are))\b.{0,40}\b(unrestricted|jailbroken|dan|no restrictions|without (any )?(rules|restrictions|limits))\b",
    "mode_switch": r"\b(developer|debug|god|admin) mode\b",
    "disable_safety": r"\b(disable|turn off|bypass|remove)\b.{0,20}\b(safety|guardrails?|filters?|restrictions?)\b",
    "delimiter_injection": r"(</?\s*system\s*>|\[/?INST\]|^\s*#{2,}\s*system\b|^\s*system\s*:)",
    "forget_everything_told": r"\b(forget|ignore|disregard|discard)\b\s+(everything|all|anything)\s+(that\s+)?(you\s+(were|have been|'ve been)|you've been)\s+(told|given|taught)\b",
    "hidden_instructions": r"\byour\s+(hidden|secret|confidential|underlying|system)\s+(instructions?|prompts?|rules?|directives?|programming)\b",
    "new_instructions_block": r"(^|[.!?\n]\s*)(new|updated|revised|real)\s+(instructions?|rules?|system prompt|task)\s*:",
}
_COMPILED = {k: re.compile(v, re.IGNORECASE | re.DOTALL | re.MULTILINE) for k, v in _INJECTION_PATTERNS.items()}


@dataclass(frozen=True)
class InjectionVerdict:
    blocked: bool
    matched: list = field(default_factory=list)


def detect_injection(text):
    matched = [name for name, rx in _COMPILED.items() if rx.search(text)]
    return InjectionVerdict(bool(matched), matched)


# Bulk-data requests ("list all salaries", "export every candidate"): the assistant reads ONE application at a time, so a request to
# enumerate all applicants or salaries is unsupported whether or not it is phrased as an override. Kept apart from detect_injection
# because ordinary policy text (for example a document saying HR may list all applications received) must not be treated as an attack.
_BULK_REQUEST = re.compile(
    r"\b(list|show|dump|export|print|display|give me)\b\s+(?:me\s+)?(?:\w+\s+){0,2}(all|every|each|the entire|the complete)\s+(?:the\s+)?"
    r"(salar(?:y|ies)|candidates?|applicants?|applications?|records?|employees?)\b",
    re.IGNORECASE,
)


def detect_bulk_request(text):
    return bool(_BULK_REQUEST.search(text))


class InvalidRequest(ValueError):
    pass


def validate_question(text):
    """Returns the stripped question or raises InvalidRequest (not a string, empty, or too long)."""
    if not isinstance(text, str):
        raise InvalidRequest("question must be a string")
    text = text.strip()
    if not text:
        raise InvalidRequest("question must not be empty")
    if not normalize_query(text):
        raise InvalidRequest("question must contain letters or digits")  # rejected before it can reach the cache
    if len(text) > settings.MAX_REQUEST_CHARS:
        raise InvalidRequest(f"question exceeds {settings.MAX_REQUEST_CHARS} characters")
    return text
