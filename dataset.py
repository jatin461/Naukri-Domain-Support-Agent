"""Deterministic generator for the fabricated JOB_APPLICATIONS dataset.

Run `PYTHONPATH=. .venv/bin/python dataset.py` to validate, print the distribution and rewrite
data/generated_job_applications.json.

Generation is a plain seeded weighted draw. If validation ever fails, the seed or the weights
below are changed and the dataset regenerated; individual records are never edited.
"""
import json
import random
from collections import Counter
from pathlib import Path

SEED = 42
N_RECORDS = 60

CATEGORY_WEIGHTS = {
    "Software Engineer": 0.28,
    "Data Analyst": 0.20,
    "Product Manager": 0.16,
    "HR Executive": 0.16,
    "Sales Associate": 0.20,
}
STATUS_WEIGHTS = {
    "Applied": 0.25,
    "Screening": 0.25,
    "Interview Scheduled": 0.20,
    "Offered": 0.10,
    "Rejected": 0.20,
}
FLAG_PROBABILITY = 0.18

SALARY_MIN_INR = 400_000
SALARY_MAX_INR = 3_500_000
SALARY_RATIONALE = (
    "Annual expected salaries of INR 400,000 to 3,500,000 run from entry-level sales and HR "
    "roles up to senior software and product roles in the Indian market, so one range is "
    "realistic for all five categories."
)

DAYS_MIN, DAYS_MAX = 0, 30
MIN_PER_CATEGORY = 3
MIN_PER_STATUS = 1
FLAG_PCT_RANGE = (10.0, 30.0)

SNAPSHOT_PATH = Path(__file__).parent / "data" / "generated_job_applications.json"
FIELDS = (
    "record_id",
    "category",
    "status",
    "expected_salary_inr",
    "days_since_created",
    "flagged_priority_review",
)


class DatasetValidationError(ValueError):
    pass


def generate_job_applications(seed=SEED, n=N_RECORDS):
    rng = random.Random(seed)
    categories, cat_w = zip(*CATEGORY_WEIGHTS.items())
    statuses, stat_w = zip(*STATUS_WEIGHTS.items())
    records = []
    for i in range(1, n + 1):
        records.append(
            {
                "record_id": f"APP-{i:04d}",
                "category": rng.choices(categories, weights=cat_w)[0],
                "status": rng.choices(statuses, weights=stat_w)[0],
                "expected_salary_inr": rng.randint(SALARY_MIN_INR, SALARY_MAX_INR),
                "days_since_created": rng.randint(DAYS_MIN, DAYS_MAX),
                "flagged_priority_review": rng.choices(
                    [True, False], weights=[FLAG_PROBABILITY, 1 - FLAG_PROBABILITY]
                )[0],
            }
        )
    return records


def flagged_percentage(records):
    return 100.0 * sum(r["flagged_priority_review"] for r in records) / len(records)


def validate_job_applications(records):
    """Raise DatasetValidationError listing every violated constraint."""
    errors = []
    if len(records) < 40:
        errors.append(f"need at least 40 records, got {len(records)}")

    for r in records:
        rid = r.get("record_id")
        if set(r) != set(FIELDS):
            errors.append(f"{rid}: fields {sorted(r)} != {sorted(FIELDS)}")
            continue
        if not isinstance(rid, str):
            errors.append(f"{rid!r}: record_id must be str")
        if r["category"] not in CATEGORY_WEIGHTS:
            errors.append(f"{rid}: unknown category {r['category']!r}")
        if r["status"] not in STATUS_WEIGHTS:
            errors.append(f"{rid}: unknown status {r['status']!r}")
        sal = r["expected_salary_inr"]
        if type(sal) is not int or not SALARY_MIN_INR <= sal <= SALARY_MAX_INR:
            errors.append(f"{rid}: salary {sal!r} outside {SALARY_MIN_INR}-{SALARY_MAX_INR}")
        d = r["days_since_created"]
        if type(d) is not int or not DAYS_MIN <= d <= DAYS_MAX:
            errors.append(f"{rid}: days_since_created {d!r} not an int in [{DAYS_MIN}, {DAYS_MAX}]")
        if type(r["flagged_priority_review"]) is not bool:
            errors.append(f"{rid}: flagged_priority_review must be bool")

    ids = [r.get("record_id") for r in records]
    if len(set(ids)) != len(ids):
        errors.append("record_id values are not unique")

    cat_counts = Counter(r.get("category") for r in records)
    for cat in CATEGORY_WEIGHTS:
        if cat_counts[cat] < MIN_PER_CATEGORY:
            errors.append(f"category {cat!r} has {cat_counts[cat]} records, need {MIN_PER_CATEGORY}")
    stat_counts = Counter(r.get("status") for r in records)
    for st in STATUS_WEIGHTS:
        if stat_counts[st] < MIN_PER_STATUS:
            errors.append(f"status {st!r} has {stat_counts[st]} records, need {MIN_PER_STATUS}")

    if records and all(isinstance(r.get("flagged_priority_review"), bool) for r in records):
        pct = flagged_percentage(records)
        lo, hi = FLAG_PCT_RANGE
        if not lo <= pct <= hi:
            errors.append(f"flagged percentage {pct:.2f} outside [{lo}, {hi}]")

    if errors:
        raise DatasetValidationError("; ".join(errors))


def format_distribution(records):
    cat = Counter(r["category"] for r in records)
    stat = Counter(r["status"] for r in records)
    flagged = sum(r["flagged_priority_review"] for r in records)
    lines = [
        f"seed={SEED} records={len(records)}",
        f"flag_probability={FLAG_PROBABILITY} salary_range_inr={SALARY_MIN_INR}-{SALARY_MAX_INR}",
        "category counts:",
        *(f"  {c}: {cat[c]}" for c in CATEGORY_WEIGHTS),
        "status counts:",
        *(f"  {s}: {stat[s]}" for s in STATUS_WEIGHTS),
        f"flagged_priority_review: {flagged}/{len(records)} = {flagged_percentage(records):.2f}%",
    ]
    return "\n".join(lines)


def write_snapshot(records, path=SNAPSHOT_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


JOB_APPLICATIONS = generate_job_applications()


def main():
    validate_job_applications(JOB_APPLICATIONS)
    write_snapshot(JOB_APPLICATIONS)
    print(format_distribution(JOB_APPLICATIONS))
    print("validation: OK")


if __name__ == "__main__":
    main()
