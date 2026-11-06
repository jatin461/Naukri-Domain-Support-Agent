"""Document-level precision and recall with visible per-query arithmetic."""
from src.config import settings
from src.rag.retrieval import parent_doc_ids, retrieve


def precision_recall(retrieved_doc_ids, relevant_doc_ids):
    retrieved = list(dict.fromkeys(retrieved_doc_ids))  # deduplicate parent IDs before scoring
    relevant = set(relevant_doc_ids)
    hits = len([d for d in retrieved if d in relevant])
    return {
        "retrieved_docs": retrieved,
        "relevant_docs": sorted(relevant),
        "hits": hits,
        "n_retrieved": len(retrieved),
        "n_relevant": len(relevant),
        "precision": hits / len(retrieved) if retrieved else 0.0,
        "recall": hits / len(relevant) if relevant else 0.0,
    }


def arithmetic(row):
    return (
        f"P = {row['hits']}/{row['n_retrieved']} = {row['precision']:.3f}; "
        f"R = {row['hits']}/{row['n_relevant']} = {row['recall']:.3f}"
    )


def evaluate_collection(collection, queries, k=settings.TOP_K):
    rows = []
    for q in queries:
        chunks = retrieve(collection, q["query"], k)
        row = precision_recall(parent_doc_ids(chunks), q["relevant"])
        row.update(query=q["query"], chunk_ids=[c.chunk_id for c in chunks])
        rows.append(row)
    return rows


def aggregate(rows):
    n = len(rows)
    return {
        "mean_precision": sum(r["precision"] for r in rows) / n,
        "mean_recall": sum(r["recall"] for r in rows) / n,
        "queries": n,
    }
