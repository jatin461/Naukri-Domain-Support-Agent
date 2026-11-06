from dataclasses import dataclass

from src.config import settings
from src.rag.embeddings import embed


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    similarity: float  # cosine similarity = 1 - Chroma cosine distance


def retrieve(collection, query, k=settings.TOP_K):
    res = collection.query(query_embeddings=embed([query]), n_results=k, include=["documents", "metadatas", "distances"])
    return [
        RetrievedChunk(cid, meta["doc_id"], meta["title"], text, 1.0 - dist)
        for cid, text, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def parent_doc_ids(chunks):
    """Parent document IDs in rank order, deduplicated."""
    return list(dict.fromkeys(c.doc_id for c in chunks))
