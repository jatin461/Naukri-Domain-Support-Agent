import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.schemas.crew_response import CrewResponse

DOC_ID_PATTERN = r"^[a-z0-9][a-z0-9_]{2,63}$"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="Support question")
    session_id: str | None = Field(default=None, max_length=64, description="Conversation session; omit for a new one")

    @field_validator("session_id")
    @classmethod
    def _session_id_chars(cls, v):
        if v is not None and not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", v):
            raise ValueError("session_id may contain only letters, digits and _ . : -")
        return v


class BudgetInfo(BaseModel):
    allowed: bool
    estimated_tokens: int
    limit: int
    simulated_cost_usd: float


class AskResponse(BaseModel):
    trace_id: str
    session_id: str
    status: Literal["answered", "fallback", "clarification", "rejected", "error"]
    answer: str
    response: CrewResponse | None = None
    sources: list[str] = Field(default_factory=list)
    fallback: bool = False
    rejection_reason: str | None = None
    budget: BudgetInfo | None = None
    duration_ms: float


class AddDocumentRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    doc_id: str | None = Field(default=None, pattern=DOC_ID_PATTERN, description="Optional stable ID; assigned when omitted")


class AddDocumentResponse(BaseModel):
    trace_id: str
    doc_id: str
    title: str
    chunks_indexed: int
    collection: str
    collection_count: int
    phone_numbers_masked: int = 0


class ErrorResponse(BaseModel):
    trace_id: str
    error: str
    detail: str | None = None
