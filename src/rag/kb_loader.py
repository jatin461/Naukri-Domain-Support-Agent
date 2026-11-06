"""Loads the knowledge-base Markdown files into documents with stable IDs and validates them.

Document ID = file name stem (e.g. `notice_period`); title = the first `# ` heading; body = the
rest of the file. Sentence counting uses `split_sentences`, which Phase 2's sentence chunker reuses.
"""
import re
from dataclasses import dataclass
from pathlib import Path

KB_DIR = Path(__file__).resolve().parents[2] / "knowledge_base"

MIN_SENTENCES, MAX_SENTENCES = 2, 5

# Required topic (Requirements §3.3) -> stable document ID.
REQUIRED_TOPICS = {
    "Job-application eligibility criteria": "job_application_eligibility",
    "Interview-scheduling process": "interview_scheduling",
    "Offer-negotiation policy": "offer_negotiation",
    "Background-verification process": "background_verification",
    "Notice-period policy": "notice_period",
    "Referral-bonus policy": "referral_bonus",
    "Internal-transfer eligibility": "internal_transfer",
    "Probation-period policy": "probation_period",
    "Remote-work eligibility": "remote_work",
    "Diversity-hiring guidelines": "diversity_hiring",
    "Exit-interview process": "exit_interview",
    "Applicant-data-retention policy": "applicant_data_retention",
}

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


class KBValidationError(ValueError):
    pass


@dataclass(frozen=True)
class KBDocument:
    doc_id: str
    title: str
    text: str
    path: Path

    @property
    def sentences(self):
        return split_sentences(self.text)


def split_sentences(text):
    text = " ".join(text.split())
    return [s for s in _SENTENCE_BOUNDARY.split(text) if s]


def _parse(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("# "):
        raise KBValidationError(f"{path.name}: first line must be a '# Title' heading")
    return KBDocument(
        doc_id=path.stem,
        title=lines[0][2:].strip(),
        text=" ".join(" ".join(lines[1:]).split()),
        path=path,
    )


def validate_kb(docs):
    """Raise KBValidationError listing every violation."""
    errors = []
    if len(docs) < len(REQUIRED_TOPICS):
        errors.append(f"need at least {len(REQUIRED_TOPICS)} documents, got {len(docs)}")
    ids = [d.doc_id for d in docs]
    if len(set(ids)) != len(ids):
        errors.append("document IDs are not unique")
    titles = [d.title for d in docs]
    if len(set(titles)) != len(titles):
        errors.append("document titles are not unique")
    for topic, doc_id in REQUIRED_TOPICS.items():
        if doc_id not in ids:
            errors.append(f"required topic {topic!r} has no document {doc_id!r}")
    for d in docs:
        n = len(d.sentences)
        if not MIN_SENTENCES <= n <= MAX_SENTENCES:
            errors.append(f"{d.doc_id}: {n} sentences, need {MIN_SENTENCES}-{MAX_SENTENCES}")
        if not d.text.rstrip().endswith((".", "!", "?")):
            errors.append(f"{d.doc_id}: text does not end with a complete sentence")
    if errors:
        raise KBValidationError("; ".join(errors))


def load_kb(kb_dir=KB_DIR, validate=True):
    docs = [_parse(p) for p in sorted(Path(kb_dir).glob("*.md"))]
    if validate:
        validate_kb(docs)
    return docs


def format_kb_report(docs):
    lines = [f"documents: {len(docs)} (sentence range required: {MIN_SENTENCES}-{MAX_SENTENCES})", "topic -> doc_id (sentences):"]
    by_id = {d.doc_id: d for d in docs}
    for topic, doc_id in REQUIRED_TOPICS.items():
        lines.append(f"  {topic} -> {doc_id} ({len(by_id[doc_id].sentences)})")
    return "\n".join(lines)
