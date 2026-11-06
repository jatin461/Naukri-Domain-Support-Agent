"""Empirical similarity-threshold calibration."""
from src.rag.retrieval import retrieve

UNTESTED_PRESETS = (0.5, 0.6, 0.7)


class ClusterOverlap(ValueError):
    pass


def measure(collection, queries):
    rows = []
    for q in queries:
        top = retrieve(collection, q, k=1)[0]
        rows.append({"query": q, "top1_similarity": round(top.similarity, 4), "top1_doc_id": top.doc_id})
    return rows


def choose_threshold(in_scope_values, out_of_scope_values):
    """Midpoint of the gap between the two observed clusters, rounded to 3 decimals."""
    lo, hi = min(in_scope_values), max(out_of_scope_values)
    if lo <= hi:
        raise ClusterOverlap(f"in-scope minimum {lo} does not exceed out-of-scope maximum {hi}")
    threshold = round((lo + hi) / 2, 3)
    if threshold in UNTESTED_PRESETS:
        raise ClusterOverlap(f"midpoint {threshold} equals an untested preset; revisit the query set")
    assert hi < threshold < lo
    return threshold


def calibrate(collection, in_scope, out_of_scope, held_in, held_out):
    in_rows = measure(collection, in_scope)
    out_rows = measure(collection, out_of_scope)
    in_vals = [r["top1_similarity"] for r in in_rows]
    out_vals = [r["top1_similarity"] for r in out_rows]
    threshold = choose_threshold(in_vals, out_vals)
    held_in_rows = measure(collection, held_in)
    held_out_rows = measure(collection, held_out)
    for r in held_in_rows:
        r["passes"] = r["top1_similarity"] >= threshold
    for r in held_out_rows:
        r["passes"] = r["top1_similarity"] < threshold
    nearest = min(
        (abs(r["top1_similarity"] - threshold), r["query"])
        for r in in_rows + out_rows + held_in_rows + held_out_rows
    )
    return {
        "in_scope": in_rows,
        "out_of_scope": out_rows,
        "in_scope_min": min(in_vals),
        "in_scope_max": max(in_vals),
        "out_of_scope_min": min(out_vals),
        "out_of_scope_max": max(out_vals),
        "gap": round(min(in_vals) - max(out_vals), 4),
        "threshold": threshold,
        "held_out_in_scope": held_in_rows,
        "held_out_out_of_scope": held_out_rows,
        "closest_to_threshold_margin": round(nearest[0], 4),
        "closest_to_threshold_query": nearest[1],
    }
