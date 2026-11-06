"""Output-side groundedness guard (deterministic; no model judges semantic groundedness).

Evidence = text of the grounded retrieval results (+ their document titles) and the facts returned
by the lookup tool. A draft sentence is "supported" when at least GROUNDING_SUPPORT_THRESHOLD of its
content words (lower-cased alphanumeric tokens minus a short stop list) occur in the evidence.
Actions (decision D5):
  refuse  no evidence supports the question at all, yet the draft makes claims -> replaced by the fallback
  flag    some draft sentences are unsupported -> passed on, flagged, for the AutoGen reviewer to revise
  pass    every sentence is supported
"""
from dataclasses import dataclass, field

from src.config import settings
from src.guardrails.support import content_tokens, sentence_support  # noqa: F401  (re-exported)
from src.llm.mock_llm import FALLBACK_ANSWER
from src.rag.kb_loader import split_sentences

@dataclass
class GroundednessVerdict:
    action: str  # "pass" | "flag" | "refuse"
    unsupported_sentences: list = field(default_factory=list)
    sentence_scores: list = field(default_factory=list)
    reason: str = ""


def lookup_evidence_text(lookup_result):
    if not lookup_result.get("found"):
        return lookup_result.get("error", "")
    r = lookup_result
    verdict = "recommended" if r["escalation_recommended"] else "not recommended"
    return (
        f"Application {r['record_id']} is at status {r['status']} with expected salary INR {r['expected_salary_inr']:,}. "
        f"Escalation score {r['escalation_score']:.2f} against threshold {r['escalation_threshold']:.2f}: escalation {verdict}."
    )


def build_evidence(rag_results, lookup_results):
    """Evidence passages: grounded retrieved chunks (with their document titles) and lookup-tool facts."""
    grounded_rag = [r for r in rag_results if r.grounded]
    passages = [f"{c.title}: {c.text}" for r in grounded_rag for c in r.chunks]
    return passages + [lookup_evidence_text(x) for x in lookup_results]


def check_groundedness(answer, rag_results, lookup_results, threshold=settings.GROUNDING_SUPPORT_THRESHOLD):
    """`rag_results`: GroundedResult list; `lookup_results`: lookup-tool dict list."""
    evidence = build_evidence(rag_results, lookup_results)
    if not evidence:
        if answer.strip() == FALLBACK_ANSWER:
            return GroundednessVerdict("pass", reason="fallback answer makes no claim")
        return GroundednessVerdict("refuse", reason="retrieved context does not support the question")
    tokens = content_tokens(" ".join(evidence))
    sentences = split_sentences(answer)
    scores = [round(sentence_support(s, tokens), 3) for s in sentences]
    unsupported = [s for s, sc in zip(sentences, scores) if sc < threshold]
    if unsupported:
        return GroundednessVerdict("flag", unsupported, scores, f"{len(unsupported)} of {len(sentences)} sentences unsupported by the evidence")
    return GroundednessVerdict("pass", [], scores, "every sentence is supported by the evidence")
