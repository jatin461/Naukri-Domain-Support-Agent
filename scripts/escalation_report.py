"""Computes the escalation-threshold evidence from the generated dataset.

Usage: PYTHONPATH=. .venv/bin/python -m scripts.escalation_report
The justification paragraph is hand-written in evaluation/hand_written/escalation_justification.md.
"""
import json

import dataset
from src.config import settings
from src.tools.application_lookup import (
    ESCALATION_THRESHOLD, FLAG_WEIGHT, MAX_DAYS, RECENCY_WEIGHT, compute_escalation_score, percentile,
)

JUSTIFICATION = settings.PROJECT_ROOT / "evaluation" / "hand_written" / "escalation_justification.md"
PENDING = "_[justification not yet written]_"


def compute():
    recs = dataset.JOB_APPLICATIONS
    days = [r["days_since_created"] for r in recs]
    scored = [(r, compute_escalation_score(r["flagged_priority_review"], r["days_since_created"])) for r in recs]
    esc = [(r, s) for r, s in scored if s >= ESCALATION_THRESHOLD]
    p20, p50, p80 = (round(percentile(days, p), 4) for p in (20, 50, 80))
    return {
        "formula": f"{FLAG_WEIGHT:.2f} * flagged_priority_review + {RECENCY_WEIGHT:.2f} * (1 - days_since_created / {MAX_DAYS})",
        "threshold": ESCALATION_THRESHOLD,
        "records": len(recs),
        "days_percentiles": {"p20": p20, "p50": p50, "p80": p80},
        "score_min": min(s for _, s in scored),
        "score_max": max(s for _, s in scored),
        "flagged_score_min": min(s for r, s in scored if r["flagged_priority_review"]),
        "unflagged_score_max": max(s for r, s in scored if not r["flagged_priority_review"]),
        "escalated": len(esc),
        "escalated_flagged": sum(r["flagged_priority_review"] for r, _ in esc),
        "escalated_unflagged_recent": sum(not r["flagged_priority_review"] for r, _ in esc),
        "unflagged_escalate_up_to_days": max(
            d for d in range(MAX_DAYS + 1) if compute_escalation_score(False, d) >= ESCALATION_THRESHOLD
        ),
        "unflagged_total_with_days_below_p20": sum(
            (not r["flagged_priority_review"]) and r["days_since_created"] < p20 for r in recs
        ),
        "regions": [
            {"days": d, "unflagged_score": compute_escalation_score(False, d), "flagged_score": compute_escalation_score(True, d),
             "unflagged_escalates": compute_escalation_score(False, d) >= ESCALATION_THRESHOLD,
             "flagged_escalates": compute_escalation_score(True, d) >= ESCALATION_THRESHOLD}
            for d in (0, 1, 2, 3, 4, 5, 6, 7, 10, 15, 20, 23, 25, 30)
        ],
    }


def render(c):
    p = c["days_percentiles"]
    L = ["# Escalation score and threshold", "",
         f"Formula: `escalation_score = {c['formula']}`; recommended escalation when score >= **{c['threshold']}**.",
         "Newer applications score higher (they have waited longest for a first response); the priority flag adds a fixed 0.60.", "",
         f"Computed from the generated dataset ({c['records']} records, seed {dataset.SEED}):", "",
         f"- days_since_created percentiles (linear interpolation): P20 = {p['p20']:.2f}, P50 = {p['p50']:.2f}, P80 = {p['p80']:.2f}",
         f"- score range over all records: {c['score_min']:.4f} to {c['score_max']:.4f} (inside [0, 1])",
         f"- flagged records score at least {c['flagged_score_min']:.4f}; unflagged records at most {c['unflagged_score_max']:.4f}",
         f"- unflagged records escalate only when days_since_created <= {c['unflagged_escalate_up_to_days']} "
         f"(every such record is below P20 = {p['p20']:.2f}); {c['unflagged_total_with_days_below_p20']} unflagged records qualify",
         f"- escalated: {c['escalated']} of {c['records']} = {c['escalated_flagged']} flagged + {c['escalated_unflagged_recent']} unflagged-but-recent", "",
         "Decision regions at the chosen threshold:", "",
         "| days_since_created | unflagged score | unflagged escalates | flagged score | flagged escalates |", "|---|---|---|---|---|"]
    L += [f"| {r['days']} | {r['unflagged_score']:.4f} | {'yes' if r['unflagged_escalates'] else 'no'} | {r['flagged_score']:.4f} | {'yes' if r['flagged_escalates'] else 'no'} |" for r in c["regions"]]
    L += ["", "Any threshold in (0.40, 0.60] would equal the boolean flag exactly, so the threshold is deliberately below 0.40, where both components change outcomes.", "",
          "## Justification (hand-written)", "", JUSTIFICATION.read_text().strip() if JUSTIFICATION.exists() else PENDING, ""]
    return "\n".join(L)


def main():
    c = compute()
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (settings.REPORTS_DIR / "escalation_threshold.json").write_text(json.dumps(c, indent=2) + "\n")
    (settings.REPORTS_DIR / "escalation_threshold.md").write_text(render(c))
    print(json.dumps({k: v for k, v in c.items() if k != "regions"}, indent=2))


if __name__ == "__main__":
    main()
