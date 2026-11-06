import re
from pathlib import Path

import pytest

from src.rag.kb_loader import (
    KB_DIR,
    REQUIRED_TOPICS,
    KBDocument,
    KBValidationError,
    load_kb,
    split_sentences,
    validate_kb,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def docs():
    return load_kb()


@pytest.mark.ac("P1-06")
def test_doc_count_and_sentence_range(docs):
    assert len(docs) >= 12
    for d in docs:
        assert 2 <= len(d.sentences) <= 5, d.doc_id
        assert d.text.endswith(".")


@pytest.mark.ac("P1-07")
def test_all_twelve_topics_present(docs):
    assert len(REQUIRED_TOPICS) == 12
    ids = {d.doc_id for d in docs}
    assert set(REQUIRED_TOPICS.values()) <= ids
    assert {p.stem for p in KB_DIR.glob("*.md")} >= set(REQUIRED_TOPICS.values())
    assert len({d.title for d in docs}) == len(docs)


def test_ids_are_stable_across_loads(docs):
    assert [d.doc_id for d in load_kb()] == [d.doc_id for d in docs]


def test_sentence_splitter():
    assert split_sentences("One thing. Two things! Is it three? Yes.") == [
        "One thing.",
        "Two things!",
        "Is it three?",
        "Yes.",
    ]
    assert len(split_sentences("The bonus is INR 25,000 for staff.  It is paid after 90 days.")) == 2


def _doc(i, sentences=3, title=None):
    return KBDocument(f"d{i}", title or f"T{i}", " ".join(["A sentence."] * sentences), Path("x.md"))


def test_validator_rejects_missing_topic_and_bad_counts(docs):
    with pytest.raises(KBValidationError, match="required topic"):
        validate_kb([_doc(i) for i in range(12)])
    too_long = [KBDocument(d.doc_id, d.title, d.text + " Extra one. Extra two.", d.path) for d in docs]
    with pytest.raises(KBValidationError, match="sentences"):
        validate_kb(too_long)
    one = [KBDocument(d.doc_id, d.title, "Only one sentence.", d.path) for d in docs]
    with pytest.raises(KBValidationError, match="sentences"):
        validate_kb(one)
    with pytest.raises(KBValidationError, match="at least 12"):
        validate_kb(docs[:11])


def test_loader_rejects_file_without_title(tmp_path):
    (tmp_path / "x.md").write_text("No heading. Two sentences.\n")
    with pytest.raises(KBValidationError, match="Title"):
        load_kb(tmp_path)


def _words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def _ngrams(words, n):
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


@pytest.mark.ac("X-24")
def test_kb_text_not_copied_from_specs(docs):
    sources = [*(ROOT / "specs").glob("*.md"), ROOT / "docs" / "capstone_brief.md"]
    source_grams = set()
    for p in sources:
        source_grams |= _ngrams(_words(p.read_text(encoding="utf-8")), 8)
    for d in docs:
        shared = _ngrams(_words(d.text), 8) & source_grams
        assert not shared, f"{d.doc_id} shares an 8-word run with the specs: {sorted(shared)[:1]}"
