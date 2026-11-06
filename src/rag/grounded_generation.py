"""Context-only answer generation with a similarity-based fallback.

Under MOCK_LLM the groundedness signal is the top-1 retrieval similarity only; no model judges
semantic groundedness. The prompt goes through the shared MockLLM core, whose "generation" is
extractive: the answer is built from the retrieved chunk texts and nothing else.
"""
from dataclasses import dataclass, field

from src.config import settings
from src.llm.mock_llm import FALLBACK_ANSWER, get_mock_llm
from src.rag.retrieval import parent_doc_ids, retrieve

SYSTEM_PROMPT = "Answer the question using only the context chunks below. If they do not contain the answer, say you do not know."


@dataclass
class GroundedResult:
    query: str
    answer: str
    grounded: bool
    top_similarity: float
    threshold: float
    source_doc_ids: list = field(default_factory=list)
    chunks: list = field(default_factory=list)


def build_context(chunks):
    return "\n".join(c.text for c in chunks)


def build_messages(query, chunks):
    context = "\n".join(f"[CHUNK {i} | {c.title}] {c.text}" for i, c in enumerate(chunks, 1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]


def generate_grounded_answer(query, collection, threshold, k=settings.TOP_K, llm=None):
    chunks = retrieve(collection, query, k)
    top = chunks[0].similarity if chunks else 0.0
    if not chunks or top < threshold:
        return GroundedResult(query, FALLBACK_ANSWER, False, top, threshold, [], chunks)
    answer = (llm or get_mock_llm()).complete("grounded_answer", build_messages(query, chunks))
    return GroundedResult(query, answer, True, top, threshold, parent_doc_ids(chunks), chunks)
