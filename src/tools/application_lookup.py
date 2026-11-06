"""Application-status lookup and the escalation score.

escalation_score = FLAG_WEIGHT * flagged_priority_review + RECENCY_WEIGHT * (1 - days_since_created / MAX_DAYS)

- flag component (0.60): a record marked for priority review is always worth a human look; on its
  own it contributes more than the whole recency component can.
- recency component (0.40): newer applications score higher (a fresh application is the one
  waiting longest for a first response); it falls linearly from 0.40 (created today) to 0 (30 days).
Both components lie in [0, 1] before weighting and the weights sum to 1, so the score is in [0, 1].
An application is recommended for escalation to a human when score >= ESCALATION_THRESHOLD.
The threshold's justification (computed percentiles) is in evaluation/reports/escalation_threshold.md.
The score only routes a case to a human; it never decides anything about a candidate.
"""
import json
import re

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

import dataset
from src.tools.recorder import InvocationRecorder

FLAG_WEIGHT = 0.60
RECENCY_WEIGHT = 0.40
MAX_DAYS = 30
ESCALATION_THRESHOLD = 0.33

RECORD_ID_PATTERN = r"APP-\d{1,6}"
_RECORD_ID_RE = re.compile(rf"\b{RECORD_ID_PATTERN}\b", re.IGNORECASE)


def detect_record_id(text):
    """First well-formed record ID in `text`, upper-cased, or None."""
    m = _RECORD_ID_RE.search(text or "")
    return m.group(0).upper() if m else None


def compute_escalation_score(flagged_priority_review, days_since_created):
    if type(flagged_priority_review) is not bool:
        raise ValueError("flagged_priority_review must be a bool")
    if type(days_since_created) is not int or not 0 <= days_since_created <= MAX_DAYS:
        raise ValueError(f"days_since_created must be an int in [0, {MAX_DAYS}]")
    score = FLAG_WEIGHT * flagged_priority_review + RECENCY_WEIGHT * (1 - days_since_created / MAX_DAYS)
    return round(score, 4)


def escalation_recommended(score):
    return score >= ESCALATION_THRESHOLD


def percentile(values, p):
    """Linear-interpolation percentile (same definition as numpy's default)."""
    xs = sorted(values)
    pos = (len(xs) - 1) * p / 100
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


_INDEX = {r["record_id"]: r for r in dataset.JOB_APPLICATIONS}


def check_job_application_status(record_id: str) -> dict:
    if not isinstance(record_id, str) or not record_id.strip():
        return {"found": False, "record_id": record_id, "error": "record_id must be a non-empty string."}
    rid = record_id.strip().upper()
    rec = _INDEX.get(rid)
    if rec is None:
        return {"found": False, "record_id": rid, "error": f"No application found for record ID {rid}."}
    score = compute_escalation_score(rec["flagged_priority_review"], rec["days_since_created"])
    return {
        "found": True,
        "record_id": rid,
        "status": rec["status"],
        "expected_salary_inr": rec["expected_salary_inr"],
        "escalation_score": score,
        "escalation_threshold": ESCALATION_THRESHOLD,
        "escalation_recommended": escalation_recommended(score),
        "category": rec["category"],
        "days_since_created": rec["days_since_created"],
        "flagged_priority_review": rec["flagged_priority_review"],
    }


class StatusLookupInput(BaseModel):
    record_id: str = Field(..., pattern=rf"^{RECORD_ID_PATTERN}$", description="Application record ID such as APP-0007")


class CheckJobApplicationStatusTool(BaseTool):
    """CrewAI wrapper. Wired to the Lookup Agent only (see src/governance/autonomy.py in Phase 8)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "check_job_application_status"
    description: str = "Look up a job application by record ID; returns status, expected salary and escalation score."
    args_schema: type[BaseModel] = StatusLookupInput
    recorder: InvocationRecorder = Field(default_factory=InvocationRecorder, exclude=True)

    @property
    def invocations(self):
        return self.recorder.records

    def _run(self, record_id: str) -> str:
        result = check_job_application_status(record_id)
        self.recorder.add(self.name, {"record_id": record_id}, result)
        return json.dumps(result)
