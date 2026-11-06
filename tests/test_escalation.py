import json

import numpy as np
import pytest

import dataset
from dataset import JOB_APPLICATIONS
from src.config import settings
from src.rag.kb_loader import split_sentences
from src.tools import application_lookup as al
from src.tools.application_lookup import (
    ESCALATION_THRESHOLD, check_job_application_status, compute_escalation_score, detect_record_id,
    escalation_recommended, percentile,
)

REPORT = settings.REPORTS_DIR / "escalation_threshold.json"
DAYS = [r["days_since_created"] for r in JOB_APPLICATIONS]


@pytest.mark.ac("P2-01")
@pytest.mark.ac("X-03")
def test_known_unknown_and_bounds():
    r = JOB_APPLICATIONS[6]  # APP-0007
    out = check_job_application_status("APP-0007")
    assert out["found"] and out["status"] == r["status"] and out["expected_salary_inr"] == r["expected_salary_inr"]
    assert 0 <= out["escalation_score"] <= 1
    assert check_job_application_status(" app-0007 ")["record_id"] == "APP-0007"  # normalised
    for rec in JOB_APPLICATIONS:
        out = check_job_application_status(rec["record_id"])
        assert out["found"] and 0 <= out["escalation_score"] <= 1
        assert out["escalation_recommended"] == (out["escalation_score"] >= ESCALATION_THRESHOLD)


@pytest.mark.ac("X-03")
@pytest.mark.parametrize("bad", ["APP-9999", "APP-0000", "nonsense", "APP-61", "", "   ", None, 7])
def test_unknown_id(bad):
    out = check_job_application_status(bad)
    assert out["found"] is False and "error" in out
    assert not {"status", "expected_salary_inr", "escalation_score"} & set(out)  # nothing invented


def test_record_id_detection():
    assert detect_record_id("what is the status of app-0007?") == "APP-0007"
    assert detect_record_id("APP-12 and APP-0003") == "APP-12"
    assert detect_record_id("notice period for XAPP-0007") is None
    assert detect_record_id("no id here") is None


def test_score_boundaries_and_components():
    assert compute_escalation_score(True, 0) == 1.0
    assert compute_escalation_score(False, 30) == 0.0
    assert compute_escalation_score(False, 0) == 0.4
    assert compute_escalation_score(True, 30) == 0.6
    assert compute_escalation_score(False, 15) == 0.2
    assert compute_escalation_score(True, 15) == 0.8
    for bad in [(1, 5), (True, 31), (True, -1), (False, 2.0), (None, 3)]:
        with pytest.raises(ValueError):
            compute_escalation_score(*bad)


@pytest.mark.ac("P2-02")
def test_both_components_change_outcome():
    # flag flips the decision at the same age
    old = [d for d in range(31) if not escalation_recommended(compute_escalation_score(False, d))]
    assert old and all(escalation_recommended(compute_escalation_score(True, d)) for d in old)
    # recency flips the decision at the same flag value
    outcomes = {escalation_recommended(compute_escalation_score(False, d)) for d in range(31)}
    assert outcomes == {True, False}
    assert {escalation_recommended(compute_escalation_score(True, d)) for d in range(31)} == {True}
    # and both happen to real records in the generated dataset
    scored = [(r, compute_escalation_score(r["flagged_priority_review"], r["days_since_created"])) for r in JOB_APPLICATIONS]
    flag_driven = [r for r, s in scored if r["flagged_priority_review"] and not escalation_recommended(compute_escalation_score(False, r["days_since_created"]))]
    recency_driven = [r for r, s in scored if not r["flagged_priority_review"] and escalation_recommended(s)]
    assert flag_driven and recency_driven
    # a bare boolean would escalate exactly the flagged records
    assert {r["record_id"] for r, s in scored if escalation_recommended(s)} != {r["record_id"] for r in JOB_APPLICATIONS if r["flagged_priority_review"]}
    assert not 0.40 < ESCALATION_THRESHOLD <= 0.60  # degenerate region (decision D3)


def test_percentile_matches_numpy():
    for p in (0, 20, 50, 80, 100):
        assert percentile(DAYS, p) == pytest.approx(np.percentile(DAYS, p))


@pytest.mark.ac("P2-03")
def test_percentile_computed_not_hardcoded():
    rep = json.loads(REPORT.read_text())
    p20, p80 = percentile(DAYS, 20), percentile(DAYS, 80)
    assert rep["days_percentiles"]["p20"] == pytest.approx(p20, abs=1e-3)
    assert rep["days_percentiles"]["p80"] == pytest.approx(p80, abs=1e-3)
    assert rep["threshold"] == ESCALATION_THRESHOLD
    # the threshold is tied to the computed P20: unflagged records escalate exactly when fresher than P20
    for d in range(31):
        assert escalation_recommended(compute_escalation_score(False, d)) == (d < p20)
    scored = [(r, compute_escalation_score(r["flagged_priority_review"], r["days_since_created"])) for r in JOB_APPLICATIONS]
    assert rep["escalated"] == sum(escalation_recommended(s) for _, s in scored)
    assert rep["escalated_flagged"] == sum(r["flagged_priority_review"] for r, s in scored if escalation_recommended(s))
    assert rep["escalated_unflagged_recent"] == sum(not r["flagged_priority_review"] for r, s in scored if escalation_recommended(s))
    assert rep["formula"] == "0.60 * flagged_priority_review + 0.40 * (1 - days_since_created / 30)"


@pytest.mark.ac("P2-03")
def test_justification_cites_computed_values():
    rep = json.loads(REPORT.read_text())
    text = (settings.PROJECT_ROOT / "evaluation" / "hand_written" / "escalation_justification.md").read_text().strip()
    assert len(split_sentences(text)) <= 4
    p = rep["days_percentiles"]
    for needle in (f"{p['p20']:.2f}", f"{p['p80']:.2f}", f"{ESCALATION_THRESHOLD:.2f}", str(rep["escalated"]),
                   str(rep["escalated_flagged"]), str(rep["escalated_unflagged_recent"]), f"{rep['flagged_score_min']:.4f}"):
        assert needle in text, needle
    assert text in (settings.REPORTS_DIR / "escalation_threshold.md").read_text()
