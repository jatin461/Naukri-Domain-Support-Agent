import copy
import json

import pytest

import dataset
from dataset import (
    CATEGORY_WEIGHTS,
    JOB_APPLICATIONS,
    SALARY_MAX_INR,
    SALARY_MIN_INR,
    STATUS_WEIGHTS,
    DatasetValidationError,
    flagged_percentage,
    generate_job_applications,
    validate_job_applications,
)

REQUIRED_CATEGORIES = {"Software Engineer", "Data Analyst", "Product Manager", "HR Executive", "Sales Associate"}
REQUIRED_STATUSES = {"Applied", "Screening", "Interview Scheduled", "Offered", "Rejected"}


@pytest.mark.ac("P1-01")
def test_min_records_and_determinism():
    assert len(JOB_APPLICATIONS) >= 40
    again = generate_job_applications()
    assert json.dumps(again) == json.dumps(JOB_APPLICATIONS)
    assert generate_job_applications(seed=7) != JOB_APPLICATIONS


@pytest.mark.ac("P1-02")
def test_category_coverage():
    assert set(CATEGORY_WEIGHTS) == REQUIRED_CATEGORIES
    for cat in REQUIRED_CATEGORIES:
        assert sum(r["category"] == cat for r in JOB_APPLICATIONS) >= 3


@pytest.mark.ac("P1-03")
def test_status_coverage():
    assert set(STATUS_WEIGHTS) == REQUIRED_STATUSES
    for st in REQUIRED_STATUSES:
        assert sum(r["status"] == st for r in JOB_APPLICATIONS) >= 1


@pytest.mark.ac("P1-04")
def test_priority_percentage():
    assert 10 <= flagged_percentage(JOB_APPLICATIONS) <= 30


@pytest.mark.ac("P1-04")
def test_snapshot_equals_regeneration():
    snapshot = json.loads(dataset.SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert snapshot == generate_job_applications()


@pytest.mark.ac("P1-05")
@pytest.mark.ac("X-20")
def test_field_types_and_ranges():
    ids = [r["record_id"] for r in JOB_APPLICATIONS]
    assert len(set(ids)) == len(ids)
    assert ids[0] == "APP-0001" and ids[-1] == f"APP-{len(ids):04d}"
    for r in JOB_APPLICATIONS:
        assert set(r) == set(dataset.FIELDS)
        assert type(r["expected_salary_inr"]) is int
        assert SALARY_MIN_INR <= r["expected_salary_inr"] <= SALARY_MAX_INR
        assert type(r["days_since_created"]) is int and 0 <= r["days_since_created"] <= 30
        assert type(r["flagged_priority_review"]) is bool
    assert dataset.SALARY_RATIONALE.count(".") == 1  # one sentence


def test_validator_accepts_generated_data():
    validate_job_applications(JOB_APPLICATIONS)


def _mutated(**changes):
    data = copy.deepcopy(JOB_APPLICATIONS)
    for k, v in changes.items():
        data[0][k] = v
    return data


@pytest.mark.parametrize(
    "bad",
    [
        JOB_APPLICATIONS[:39],
        _mutated(days_since_created=31),
        _mutated(days_since_created=-1),
        _mutated(days_since_created=3.0),
        _mutated(flagged_priority_review=1),
        _mutated(expected_salary_inr=SALARY_MIN_INR - 1),
        _mutated(expected_salary_inr=SALARY_MAX_INR + 1),
        _mutated(category="Janitor"),
        _mutated(status="Hired"),
        _mutated(record_id=JOB_APPLICATIONS[1]["record_id"]),
        [{**r, "flagged_priority_review": False} for r in JOB_APPLICATIONS],
        [{**r, "flagged_priority_review": True} for r in JOB_APPLICATIONS],
        [{**r, "category": "Data Analyst"} for r in JOB_APPLICATIONS],
        [{**r, "status": "Applied"} for r in JOB_APPLICATIONS],
        [{k: v for k, v in r.items() if k != "status"} for r in JOB_APPLICATIONS],
    ],
)
def test_validator_rejects_bad_data(bad):
    with pytest.raises(DatasetValidationError):
        validate_job_applications(bad)
