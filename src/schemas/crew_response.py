from typing import Literal

from pydantic import BaseModel, Field, model_validator

ResponseType = Literal["policy", "lookup", "policy_and_lookup", "no_evidence", "clarification"]


class CrewResponse(BaseModel):
    """Structured response of the crew; every crew output is validated against it."""

    answer: str = Field(min_length=1)
    response_type: ResponseType
    sources: list[str] = Field(default_factory=list)
    grounded: bool = False
    record_id: str | None = None
    status: str | None = None
    expected_salary_inr: int | None = Field(default=None, ge=0)
    escalation_score: float | None = Field(default=None, ge=0, le=1)
    escalation_recommended: bool | None = None

    @model_validator(mode="after")
    def _lookup_fields_complete(self):
        if self.status is not None and None in (self.record_id, self.expected_salary_inr, self.escalation_score, self.escalation_recommended):
            raise ValueError("a looked-up record must carry record_id, expected_salary_inr, escalation_score and escalation_recommended")
        if "lookup" in self.response_type and self.record_id is None:
            raise ValueError("lookup responses must name the record_id")
        return self


# The crew's response_format: passed to the Composer task as CrewAI's `output_pydantic` and
# re-validated in code after every `.kickoff()`.
response_format = CrewResponse
