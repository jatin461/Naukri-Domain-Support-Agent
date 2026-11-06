from pydantic import BaseModel


class Verdict(BaseModel):
    """Structured verdict of the AutoGen review team (the Final_Editor's output_content_type).

    approved=True means the draft is delivered unchanged; approved=False means final_answer is a revision
    of the draft (the reason says what was removed)."""

    approved: bool
    final_answer: str
    reason: str
