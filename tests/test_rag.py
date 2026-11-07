import json
import math
import os
import re
import subprocess
import sys

import pytest

from evaluation import rag_queries as Q
from src.config import settings
from src.rag import embeddings, vector_store
from src.rag.calibration import UNTESTED_PRESETS, ClusterOverlap, choose_threshold, measure
from src.rag.chunking import chunk_fixed, chunk_sentences
from src.rag.grounded_generation import FALLBACK_ANSWER, build_context, generate_grounded_answer
from src.rag.kb_loader import KBDocument, load_kb, split_sentences
from src.rag.retrieval import parent_doc_ids, retrieve
from src.rag.retrieval_eval import aggregate, arithmetic, evaluate_collection, precision_recall

ROOT = settings.PROJECT_ROOT
STRATEGIES = list(settings.COLLECTION_NAMES)


def _doc(text, doc_id="d1"):
    from pathlib import Path

    return KBDocument(doc_id, "Title " + doc_id, text, Path("x.md"))


@pytest.fixture(scope="module")
def docs():
    return load_kb()


@pytest.fixture(scope="module")
def built(docs, tmp_path_factory):
    client = vector_store.get_client(tmp_path_factory.mktemp("chroma"))
    return vector_store.build_collections(client, docs)


def _cols(built):
    return {s: built[s][0] for s in STRATEGIES}


# ---------- chunking ----------

WORDS_70 = " ".join(f"w{i}" for i in range(70)) + "."


@pytest.mark.ac("P1-08")
def test_fixed_chunk_overlap_and_coverage():
    chunks = chunk_fixed(_doc(WORDS_70), size=30, overlap=8)
    assert len(chunks) == 3
    assert all(len(c.text.split()) <= 30 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert a.text.split()[-8:] == b.text.split()[:8]  # non-zero overlap
    covered = {w for c in chunks for w in c.text.split()}
    assert covered == set(WORDS_70.split())
    assert [c.chunk_id for c in chunks] == ["d1::fixed::00", "d1::fixed::01", "d1::fixed::02"]
    assert all(c.doc_id == "d1" and c.title == "Title d1" and c.strategy == "fixed" for c in chunks)


@pytest.mark.ac("P1-08")
def test_fixed_chunk_edge_cases():
    assert [c.text for c in chunk_fixed(_doc("Short text."))] == ["Short text."]
    assert chunk_fixed(_doc("")) == []
    assert len(chunk_fixed(_doc("one two three four five six seven eight."), size=8, overlap=2)) == 1
    with pytest.raises(ValueError):
        chunk_fixed(_doc("a b c"), size=5, overlap=0)
    with pytest.raises(ValueError):
        chunk_fixed(_doc("a b c"), size=5, overlap=5)


@pytest.mark.ac("P1-09")
def test_sentence_chunking(docs):
    chunks = chunk_sentences(_doc("First one. Second one! Third? Fourth, with punctuation; still one."))
    assert [c.text for c in chunks] == ["First one.", "Second one!", "Third?", "Fourth, with punctuation; still one."]
    assert chunks[2].chunk_id == "d1::sent::02" and chunks[2].strategy == "sentence"
    assert chunks[0].metadata() == {"doc_id": "d1", "title": "Title d1", "strategy": "sentence", "chunk_index": 0}
    assert [c.text for c in chunk_sentences(_doc("Only one sentence."))] == ["Only one sentence."]
    assert chunk_sentences(_doc("")) == []
    for d in docs:
        assert [c.text for c in chunk_sentences(d)] == split_sentences(d.text)


def test_chunk_ids_unique_per_strategy(docs):
    for fn in (chunk_fixed, chunk_sentences):
        ids = [c.chunk_id for d in docs for c in fn(d)]
        assert len(ids) == len(set(ids))


# ---------- embeddings and Chroma ----------


@pytest.mark.ac("P1-10")
def test_embeddings_local_offline():
    assert os.environ["HF_HUB_OFFLINE"] == "1"  # sockets are also blocked by the autouse fixture
    vecs = embeddings.embed(["interview slots", "notice period"])
    assert len(vecs) == 2 and len(vecs[0]) == 384
    assert math.isclose(sum(x * x for x in vecs[0]), 1.0, abs_tol=1e-4)
    assert embeddings.embed([]) == []


def test_missing_model_gives_actionable_error(monkeypatch):
    embeddings.get_model.cache_clear()
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "sentence-transformers/does-not-exist-xyz")
    try:
        with pytest.raises(embeddings.EmbeddingModelUnavailable, match="prefetch_model"):
            embeddings.get_model()
    finally:
        embeddings.get_model.cache_clear()


def test_prefetch_script_check_mode_loads_offline():
    r = subprocess.run(
        [sys.executable, "scripts/prefetch_model.py", "--check"],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": "."}, capture_output=True, text=True, timeout=180,
    )
    assert r.returncode == 0, r.stderr[-500:]
    assert "offline load OK" in r.stdout


@pytest.mark.ac("P1-11")
def test_two_collections_use_upsert(docs, tmp_path, monkeypatch):
    calls = []
    cls = type(vector_store.get_collection(vector_store.get_client(tmp_path / "probe"), "fixed"))
    original = cls.upsert

    def spy(self, *a, **kw):
        calls.append((self.name, len(kw["ids"])))
        return original(self, *a, **kw)

    monkeypatch.setattr(cls, "upsert", spy)
    client = vector_store.get_client(tmp_path / "db")
    first = vector_store.build_collections(client, docs)
    counts = {s: first[s][0].count() for s in STRATEGIES}
    second = vector_store.build_collections(client, docs)  # re-index: must not duplicate
    assert {s: second[s][0].count() for s in STRATEGIES} == counts
    assert {name for name, _ in calls} == set(settings.COLLECTION_NAMES.values())
    assert len(calls) == 4
    names = [first[s][0].name for s in STRATEGIES]
    assert len(set(names)) == 2
    assert counts["fixed"] != counts["sentence"]
    assert first["fixed"][0].metadata["hnsw:space"] == "cosine"


@pytest.mark.ac("P1-11")
def test_both_collections_return_sensible_top1(built):
    for s, col in _cols(built).items():
        top = retrieve(col, "How much referral bonus is paid for a managerial role?", 1)[0]
        assert top.doc_id == "referral_bonus", s
        assert 0.5 < top.similarity <= 1.0
        assert retrieve(col, "How long do you keep the data of rejected applicants?", 1)[0].doc_id == "applicant_data_retention"


def test_similarity_is_cosine_of_identical_text(built):
    col = built["sentence"][0]
    text = col.get(ids=["notice_period::sent::00"])["documents"][0]
    top = retrieve(col, text, 1)[0]
    assert top.chunk_id == "notice_period::sent::00" and top.similarity > 0.99


# ---------- calibration ----------


def _report(name):
    return json.loads((settings.REPORTS_DIR / name).read_text(encoding="utf-8"))


@pytest.mark.ac("P1-12")
def test_threshold_between_clusters():
    rep = _report("similarity_calibration.json")
    for s in STRATEGIES:
        r = rep[s]
        assert len(r["in_scope"]) >= 3 and len(r["out_of_scope"]) >= 2
        lo = min(x["top1_similarity"] for x in r["in_scope"])
        hi = max(x["top1_similarity"] for x in r["out_of_scope"])
        assert hi < r["threshold"] < lo
        assert r["threshold"] not in UNTESTED_PRESETS
        assert all(x["passes"] for x in r["held_out_in_scope"] + r["held_out_out_of_scope"])
    assert settings.get_similarity_threshold() == rep[settings.RAG_STRATEGY]["threshold"]


@pytest.mark.ac("P1-12")
def test_calibration_report_matches_live_measurement(built):
    rep = _report("similarity_calibration.json")
    for s, col in _cols(built).items():
        assert [r["top1_similarity"] for r in measure(col, [q["query"] for q in Q.IN_SCOPE])] == [
            r["top1_similarity"] for r in rep[s]["in_scope"]
        ]
        assert [r["top1_similarity"] for r in measure(col, Q.OUT_OF_SCOPE)] == [
            r["top1_similarity"] for r in rep[s]["out_of_scope"]
        ]
    md = (settings.REPORTS_DIR / "similarity_calibration.md").read_text()
    assert f"{rep[settings.RAG_STRATEGY]['threshold']:.3f}" in md and "not yet written" not in md


def test_choose_threshold_rules():
    assert choose_threshold([0.6, 0.8], [0.1, 0.2]) == 0.4
    with pytest.raises(ClusterOverlap):
        choose_threshold([0.3, 0.8], [0.1, 0.4])
    with pytest.raises(ClusterOverlap):
        choose_threshold([0.6, 0.8], [0.4, 0.4])  # midpoint 0.5 is an untested preset


# ---------- grounded generation ----------


@pytest.mark.ac("P1-13")
def test_five_in_scope_grounded(built):
    assert len(Q.IN_SCOPE) >= 5
    for s, col in _cols(built).items():
        thr = settings.get_similarity_threshold(s)
        for q in Q.IN_SCOPE:
            r = generate_grounded_answer(q["query"], col, thr)
            assert r.grounded and r.top_similarity >= thr
            assert set(q["relevant"]) & set(r.source_doc_ids)
            body = r.answer.split(": ", 1)[1]
            context = build_context(r.chunks)
            assert all(c.text in context for c in r.chunks)
            body_words = set(re.findall(r"[a-z0-9]+", body.lower()))
            chunk_words = set(re.findall(r"[a-z0-9]+", " ".join(c.text for c in r.chunks).lower()))
            assert body_words == chunk_words  # nothing but retrieved text, and nothing retrieved is dropped
            context_words = set(re.findall(r"[a-z0-9]+", (context + " " + " ".join(c.title for c in r.chunks)).lower()))
            assert set(re.findall(r"[a-z0-9]+", r.answer.lower())) <= context_words | {"based", "on"}


@pytest.mark.ac("P1-13")
def test_answer_never_glues_separate_pieces_into_one_sentence(built, docs):
    # Every segment between separators must be one unbroken passage of a single retrieved document,
    # so two windows that are not adjacent in the knowledge base can never read as one sentence.
    doc_text = {d.doc_id: " ".join(d.text.split()) for d in docs}
    queries = [q["query"] for q in Q.IN_SCOPE] + ["How long do you keep the data of rejected applicants?",
                                                   "Can a new joiner work from home during probation?"]
    for s, col in _cols(built).items():
        thr = settings.get_similarity_threshold(s)
        for query in queries:
            r = generate_grounded_answer(query, col, thr)
            body = r.answer.split(": ", 1)[1]
            segments = [seg.strip() for seg in body.split("…") if seg.strip()]
            assert segments
            for seg in segments:
                assert any(seg in doc_text[d] for d in r.source_doc_ids), (s, query, seg)


def test_join_pieces_marks_breaks_and_adds_no_words():
    from src.llm.mock_llm import join_pieces

    assert join_pieces(["One. Two."]) == "One. Two."
    assert join_pieces(["the end of one.", "Start of another"]) == "… the end of one. … Start of another …"
    assert re.findall(r"[a-z]+", join_pieces(["a b.", "c d"]).lower()) == ["a", "b", "c", "d"]


@pytest.mark.ac("P1-14")
def test_out_of_scope_fallback(built):
    for s, col in _cols(built).items():
        thr = settings.get_similarity_threshold(s)
        for q in [*Q.OUT_OF_SCOPE, *Q.HELD_OUT_OUT_OF_SCOPE]:
            r = generate_grounded_answer(q, col, thr)
            assert not r.grounded and r.answer == FALLBACK_ANSWER and r.source_doc_ids == []
            assert r.top_similarity < thr


# ---------- precision / recall ----------


@pytest.mark.ac("P1-15")
def test_precision_recall_math():
    row = precision_recall(["a", "a", "b", "c"], ["a", "c", "d"])  # duplicates collapse: retrieved = a, b, c
    assert (row["hits"], row["n_retrieved"], row["n_relevant"]) == (2, 3, 3)
    assert math.isclose(row["precision"], 2 / 3) and math.isclose(row["recall"], 2 / 3)
    assert arithmetic(row) == "P = 2/3 = 0.667; R = 2/3 = 0.667"
    empty = precision_recall([], ["a"])
    assert empty["precision"] == 0.0 and empty["recall"] == 0.0
    assert aggregate([row, empty])["mean_precision"] == pytest.approx(1 / 3)


@pytest.mark.ac("P1-15")
def test_comparison_report_matches_live_scores(built):
    rep = _report("chunking_comparison.json")
    for s, col in _cols(built).items():
        rows = evaluate_collection(col, Q.IN_SCOPE)
        assert [(r["hits"], r["n_retrieved"], r["n_relevant"]) for r in rows] == [
            (r["hits"], r["n_retrieved"], r["n_relevant"]) for r in rep[s]["rows"]
        ]
        assert aggregate(rows) == rep[s]["aggregate"]
    assert len(Q.IN_SCOPE) >= 5
    assert [r["query"] for r in rep["fixed"]["rows"]] == [r["query"] for r in rep["sentence"]["rows"]]


def test_harder_multi_doc_set_separates_the_strategies(built):
    rep = _report("chunking_comparison.json")
    assert len(Q.HARD_MULTI_DOC) >= 2 and all(len(q["relevant"]) == 3 for q in Q.HARD_MULTI_DOC)
    for s, col in _cols(built).items():
        rows = evaluate_collection(col, Q.HARD_MULTI_DOC)
        assert aggregate(rows) == rep[s]["hard_aggregate"]
        assert aggregate(evaluate_collection(col, Q.IN_SCOPE + Q.HARD_MULTI_DOC)) == rep[s]["combined_aggregate"]
    assert rep["fixed"]["hard_aggregate"] != rep["sentence"]["hard_aggregate"]  # the harder set actually discriminates
    text = (ROOT / "evaluation" / "hand_written" / "chunking_recommendation.md").read_text()
    for s in STRATEGIES:
        a = rep[s]["hard_aggregate"]
        assert f"{a['mean_precision']:.3f}" in text and f"{a['mean_recall']:.3f}" in text
    md = (settings.REPORTS_DIR / "chunking_comparison.md").read_text()
    assert len(re.findall(r"\| H\d \|.*P = \d+/\d+ = [\d.]+; R = \d+/\d+ = [\d.]+", md)) == 2 * len(Q.HARD_MULTI_DOC)


@pytest.mark.ac("P1-16")
def test_report_shows_per_query_arithmetic():
    md = (settings.REPORTS_DIR / "chunking_comparison.md").read_text()
    for s in STRATEGIES:
        section = md.split(f"({s}),", 1)[1].split("\n## ", 1)[0]
        rows = re.findall(r"P = (\d+)/(\d+) = [\d.]+; R = (\d+)/(\d+) = [\d.]+", section)
        assert len(rows) >= 5


@pytest.mark.ac("P1-17")
def test_recommendation_cites_numbers():
    rep = _report("chunking_comparison.json")
    cal = _report("similarity_calibration.json")
    text = (ROOT / "evaluation" / "hand_written" / "chunking_recommendation.md").read_text().strip()
    assert 2 <= len(split_sentences(text)) <= 3
    for s in STRATEGIES:
        a = rep[s]["aggregate"]
        assert f"{a['mean_precision']:.3f}" in text and f"{a['mean_recall']:.3f}" in text
    for s in STRATEGIES:
        assert f"{cal[s]['threshold']:.3f}" in text
        assert f"{cal[s]['closest_to_threshold_margin']:.4f}" in text
    chosen = {"fixed": "fixed-size", "sentence": "sentence-based"}[settings.RAG_STRATEGY]
    assert f"deploying the {chosen} collection" in text
    assert text in (settings.REPORTS_DIR / "chunking_comparison.md").read_text()


# ---------- late writes stay searchable (Chroma quirk: read after bulk load, then a small upsert) ----------


def test_chunk_added_after_a_read_is_searchable(docs, tmp_path):
    from pathlib import Path

    from src.rag.chunking import chunk_documents, chunk_fixed
    from src.rag.vector_store import IndexNotSearchable, upsert_chunks  # noqa: F401

    text = "Employees wear smart casual clothes on weekdays. Formal wear is needed only for client visits."
    new = chunk_fixed(KBDocument("added_dress", "Dress Code", text, Path("x")))
    for i in range(8):  # without the collection settings about 1 build in 5 loses the new chunk
        col = vector_store.get_collection(vector_store.get_client(tmp_path / f"c{i}"), "fixed")
        upsert_chunks(col, chunk_documents(docs, "fixed"))
        col.query(query_embeddings=embeddings.embed(["warm-up read"]), n_results=1)
        upsert_chunks(col, new, verify=True)
        top = retrieve(col, "What is the dress code for employees on weekdays?", 3)
        assert new[0].chunk_id in [c.chunk_id for c in top], i


def test_verify_repairs_a_dropped_write_and_fails_loudly_when_it_cannot(docs):
    from pathlib import Path

    from src.rag.chunking import chunk_fixed
    from src.rag.vector_store import IndexNotSearchable, upsert_chunks

    class Flaky:
        """Stores ids but 'forgets' to index the first write, like the quirk."""

        def __init__(self, drop_writes):
            self.drop, self.upserts, self.ids = drop_writes, 0, []

        def upsert(self, ids, embeddings, documents, metadatas):
            self.upserts += 1
            if self.upserts > self.drop:
                self.ids = list(ids)

        def count(self):
            return 5

        def query(self, query_embeddings, n_results):
            return {"ids": [self.ids]}

    chunk = chunk_fixed(KBDocument("d", "T", "some words here for a chunk", Path("x")))
    col = Flaky(drop_writes=1)
    assert upsert_chunks(col, chunk, verify=True) == 1 and col.upserts == 2  # repaired by a re-upsert
    with pytest.raises(IndexNotSearchable):
        upsert_chunks(Flaky(drop_writes=99), chunk, verify=True)
