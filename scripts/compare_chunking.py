"""Document-level precision/recall for both collections on the same queries.

Usage: PYTHONPATH=. .venv/bin/python -m scripts.compare_chunking
The recommendation is hand-written in evaluation/hand_written/chunking_recommendation.md.
"""
import json
import tempfile

from evaluation import rag_queries as Q
from src.config import settings
from src.rag.kb_loader import load_kb
from src.rag.retrieval_eval import aggregate, arithmetic, evaluate_collection
from src.rag.vector_store import build_collections, get_client

SENSITIVITY_K = (1, 2, 3, 5)  # primary result uses settings.TOP_K; the others only show how stable it is
RECOMMENDATION = settings.PROJECT_ROOT / "evaluation" / "hand_written" / "chunking_recommendation.md"
PENDING = "_[recommendation not yet written]_"


def run():
    with tempfile.TemporaryDirectory() as tmp:
        built = build_collections(get_client(tmp), load_kb())
        out = {}
        for s, (col, n_chunks) in built.items():
            rows = evaluate_collection(col, Q.IN_SCOPE)
            sens = {k: aggregate(evaluate_collection(col, Q.IN_SCOPE, k)) for k in SENSITIVITY_K}
            hard = evaluate_collection(col, Q.HARD_MULTI_DOC)
            hard_sens = {k: aggregate(evaluate_collection(col, Q.HARD_MULTI_DOC, k)) for k in SENSITIVITY_K}
            out[s] = {"n_chunks": n_chunks, "rows": rows, "aggregate": aggregate(rows), "k_sensitivity": sens,
                      "hard_rows": hard, "hard_aggregate": aggregate(hard), "hard_k_sensitivity": hard_sens,
                      "combined_aggregate": aggregate(rows + hard)}
        return out


def render(result):
    L = ["# Chunking strategy comparison (document-level precision and recall)", "",
         f"Top-k = {settings.TOP_K} chunks per query; retrieved chunks are mapped to parent documents and deduplicated before scoring.",
         "Precision = relevant documents retrieved / distinct documents retrieved. Recall = relevant documents retrieved / relevant documents declared.",
         f"Fixed strategy: {settings.FIXED_CHUNK_WORDS}-word windows, {settings.FIXED_CHUNK_OVERLAP_WORDS}-word overlap. Sentence strategy: one chunk per sentence.", ""]
    for s, r in result.items():
        L += [f"## {settings.COLLECTION_NAMES[s]} ({s}), {r['n_chunks']} chunks", "",
              "| # | Query | Relevant (declared) | Retrieved chunks | Retrieved docs (deduplicated) | Arithmetic |", "|---|---|---|---|---|---|"]
        for i, x in enumerate(r["rows"], 1):
            L.append(f"| {i} | {x['query']} | {', '.join(x['relevant_docs'])} | {', '.join(x['chunk_ids'])} | {', '.join(x['retrieved_docs'])} | {arithmetic(x)} |")
        a = r["aggregate"]
        L += ["", f"Mean precision = {a['mean_precision']:.3f}; mean recall = {a['mean_recall']:.3f} over {a['queries']} queries.", ""]
    f, se = result["fixed"]["aggregate"], result["sentence"]["aggregate"]
    L += ["## Summary", "", "| Strategy | Mean precision | Mean recall |", "|---|---|---|",
          f"| fixed-size with overlap | {f['mean_precision']:.3f} | {f['mean_recall']:.3f} |",
          f"| sentence-based | {se['mean_precision']:.3f} | {se['mean_recall']:.3f} |", "",
          "## Sensitivity to top-k (same queries; primary result above uses k = " + str(settings.TOP_K) + ")", "",
          "| k | fixed precision | fixed recall | sentence precision | sentence recall |", "|---|---|---|---|---|",
          *[f"| {k} | {result['fixed']['k_sensitivity'][k]['mean_precision']:.3f} | {result['fixed']['k_sensitivity'][k]['mean_recall']:.3f} | {result['sentence']['k_sensitivity'][k]['mean_precision']:.3f} | {result['sentence']['k_sensitivity'][k]['mean_recall']:.3f} |" for k in SENSITIVITY_K], "",
          "## Harder multi-document set (each query spans three documents)", "",
          "The six shared queries above are the Task 4 queries and tie. These three extra queries each need evidence from three documents, "
          "so with top-k = 3 full recall requires every retrieved chunk to come from a different relevant document.", ""]
    for s, r in result.items():
        L += [f"### {settings.COLLECTION_NAMES[s]} [{s}], harder set", "",
              "| # | Query | Relevant (declared) | Retrieved chunks | Retrieved docs (deduplicated) | Arithmetic |", "|---|---|---|---|---|---|"]
        for i, x in enumerate(r["hard_rows"], 1):
            L.append(f"| H{i} | {x['query']} | {', '.join(x['relevant_docs'])} | {', '.join(x['chunk_ids'])} | {', '.join(x['retrieved_docs'])} | {arithmetic(x)} |")
        a = r["hard_aggregate"]
        L += ["", f"Mean precision = {a['mean_precision']:.3f}; mean recall = {a['mean_recall']:.3f} over {a['queries']} queries.", ""]
    fh, sh = result["fixed"], result["sentence"]
    L += ["### All results side by side", "", "| Query set | fixed precision | fixed recall | sentence precision | sentence recall |", "|---|---|---|---|---|",
          *[f"| {name} | {fh[key]['mean_precision']:.3f} | {fh[key]['mean_recall']:.3f} | {sh[key]['mean_precision']:.3f} | {sh[key]['mean_recall']:.3f} |"
            for name, key in (("6 shared queries", "aggregate"), ("3 harder queries", "hard_aggregate"), ("all 9 queries", "combined_aggregate"))], "",
          "Harder set, sensitivity to top-k:", "",
          "| k | fixed precision | fixed recall | sentence precision | sentence recall |", "|---|---|---|---|---|",
          *[f"| {k} | {fh['hard_k_sensitivity'][k]['mean_precision']:.3f} | {fh['hard_k_sensitivity'][k]['mean_recall']:.3f} | {sh['hard_k_sensitivity'][k]['mean_precision']:.3f} | {sh['hard_k_sensitivity'][k]['mean_recall']:.3f} |" for k in SENSITIVITY_K], "",
          "## Recommendation (hand-written)", "", RECOMMENDATION.read_text().strip() if RECOMMENDATION.exists() else PENDING, ""]
    return "\n".join(L)


def main():
    result = run()
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (settings.REPORTS_DIR / "chunking_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    (settings.REPORTS_DIR / "chunking_comparison.md").write_text(render(result))
    for s, r in result.items():
        for key in ("aggregate", "hard_aggregate", "combined_aggregate"):
            print(s, key, {k: round(v, 3) if isinstance(v, float) else v for k, v in r[key].items()})
        print(s, "hard k", {k: (round(v["mean_precision"], 3), round(v["mean_recall"], 3)) for k, v in r["hard_k_sensitivity"].items()})


if __name__ == "__main__":
    main()
