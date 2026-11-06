"""Two chunkers over the same documents. Every chunk keeps its parent document metadata."""
from dataclasses import dataclass

from src.config import settings
from src.rag.kb_loader import split_sentences


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    strategy: str
    index: int
    text: str

    def metadata(self):
        return {"doc_id": self.doc_id, "title": self.title, "strategy": self.strategy, "chunk_index": self.index}


def chunk_fixed(doc, size=settings.FIXED_CHUNK_WORDS, overlap=settings.FIXED_CHUNK_OVERLAP_WORDS):
    """Fixed-size windows of `size` words; each window starts `size - overlap` words after the last."""
    if not 0 < overlap < size:
        raise ValueError("need 0 < overlap < size")
    words = doc.text.split()
    step = size - overlap
    chunks = []
    for i, start in enumerate(range(0, max(len(words) - overlap, 1), step)):
        window = words[start : start + size]
        if not window:
            break
        chunks.append(Chunk(f"{doc.doc_id}::fixed::{i:02d}", doc.doc_id, doc.title, "fixed", i, " ".join(window)))
    return chunks


def chunk_sentences(doc):
    """One chunk per sentence."""
    return [
        Chunk(f"{doc.doc_id}::sent::{i:02d}", doc.doc_id, doc.title, "sentence", i, s)
        for i, s in enumerate(split_sentences(doc.text))
    ]


CHUNKERS = {"fixed": chunk_fixed, "sentence": chunk_sentences}


def chunk_documents(docs, strategy):
    return [c for d in docs for c in CHUNKERS[strategy](d)]
