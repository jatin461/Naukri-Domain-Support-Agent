"""Measures top-1 cosine similarity on both collections and writes the calibration report.

Usage: PYTHONPATH=. .venv/bin/python -m scripts.calibrate_threshold
The rationale paragraph is hand-written in evaluation/hand_written/threshold_rationale.md.
"""
import json
import tempfile

from evaluation import rag_queries as Q
from src.config import settings
from src.rag.calibration import calibrate
from src.rag.kb_loader import load_kb
from src.rag.vector_store import build_collections, get_client

RATIONALE = settings.PROJECT_ROOT / "evaluation" / "hand_written" / "threshold_rationale.md"
PENDING = "_[rationale not yet written]_"


def run():
    with tempfile.TemporaryDirectory() as tmp:
        built = build_collections(get_client(tmp), load_kb())
        queries_in = [q["query"] for q in Q.IN_SCOPE]
        return {
            s: calibrate(col, queries_in, Q.OUT_OF_SCOPE, [q["query"] for q in Q.HELD_OUT_IN_SCOPE], Q.HELD_OUT_OUT_OF_SCOPE)
            for s, (col, _) in built.items()
        }


def render(result):
    L = ["# Similarity calibration", "",
         "Metric: top-1 cosine similarity (1 - Chroma cosine distance), model `" + settings.EMBEDDING_MODEL + "`, top-k chunks retrieved per query.",
         "Under `MOCK_LLM` this similarity is the only groundedness signal; no model judges semantic groundedness.", ""]
    for s, r in result.items():
        L += [f"## Collection: {settings.COLLECTION_NAMES[s]} ({s})", "", "| Set | Query | Top-1 similarity | Top-1 doc |", "|---|---|---|---|"]
        for tag, key in (("in-scope", "in_scope"), ("out-of-scope", "out_of_scope")):
            L += [f"| {tag} | {x['query']} | {x['top1_similarity']:.4f} | {x['top1_doc_id']} |" for x in r[key]]
        L += ["",
              f"- In-scope cluster: min {r['in_scope_min']:.4f}, max {r['in_scope_max']:.4f}",
              f"- Out-of-scope cluster: min {r['out_of_scope_min']:.4f}, max {r['out_of_scope_max']:.4f}",
              f"- Gap between clusters: {r['gap']:.4f}",
              f"- **Chosen threshold (midpoint of gap): {r['threshold']:.3f}**", "",
              f"- Closest measured query to the threshold (all 12 queries): {r['closest_to_threshold_margin']:.4f} away (\"{r['closest_to_threshold_query']}\")", "",
              "Held-out check (not used to choose the threshold):", "",
              "| Set | Query | Top-1 similarity | Behaves correctly at threshold |", "|---|---|---|---|"]
        for tag, key in (("in-scope", "held_out_in_scope"), ("out-of-scope", "held_out_out_of_scope")):
            L += [f"| {tag} | {x['query']} | {x['top1_similarity']:.4f} | {'yes' if x['passes'] else 'NO'} |" for x in r[key]]
        L.append("")
    L += ["## Rationale (hand-written)", "", RATIONALE.read_text().strip() if RATIONALE.exists() else PENDING, ""]
    return "\n".join(L)


def main():
    result = run()
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (settings.REPORTS_DIR / "similarity_calibration.json").write_text(json.dumps(result, indent=2) + "\n")
    (settings.REPORTS_DIR / "similarity_calibration.md").write_text(render(result))
    for s, r in result.items():
        print(f"{s}: in-scope min {r['in_scope_min']} | out-of-scope max {r['out_of_scope_max']} | threshold {r['threshold']}")


if __name__ == "__main__":
    main()
