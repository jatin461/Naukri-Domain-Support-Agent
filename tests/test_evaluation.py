import json

import pytest

import dataset
from evaluation.evaluation_dataset import EVALUATION_QUERIES
from evaluation.judge import Judge, JudgeScores, RUBRIC, SCALE
from evaluation.run_evaluation import FINDINGS, METRICS, build_report, render
from src.config import settings
from src.crew.routing import has_application_intent, has_policy_hint, route
from src.llm.mock_llm import FALLBACK_ANSWER, MockLLM
from src.rag.kb_loader import REQUIRED_TOPICS, load_kb, split_sentences

REPORT_JSON = settings.REPORTS_DIR / "15_query_results.json"
REPORT_MD = settings.REPORTS_DIR / "15_query_results.md"
KB = {d.doc_id: d.text.lower() for d in load_kb()}
BY_ID = {q["id"]: q for q in EVALUATION_QUERIES}


@pytest.fixture(scope="module")
def report():
    return build_report()


@pytest.mark.ac("P3-07")
def test_fifteen_queries_cover_twelve_topics_and_edges():
    assert len(EVALUATION_QUERIES) == 15
    assert len({q["id"] for q in EVALUATION_QUERIES}) == 15 and len({q["query"] for q in EVALUATION_QUERIES}) == 15
    assert set(REQUIRED_TOPICS.values()) <= {q["expected_doc_id"] for q in EVALUATION_QUERIES}
    assert sum(q["kind"] in ("out_of_scope", "injection", "edge") for q in EVALUATION_QUERIES) >= 2
    assert {q["kind"] for q in EVALUATION_QUERIES} == {"policy", "status", "out_of_scope", "injection"}


def test_expectations_are_declared_independently_and_consistent_with_the_kb():
    for q in EVALUATION_QUERIES:
        if q["kind"] == "policy":
            for fact in q["required_facts"]:
                assert fact.lower() in KB[q["expected_doc_id"]], (q["id"], fact)  # the declared fact really is in the KB
        for claim in q["forbidden_claims"]:
            if q["kind"] == "policy":
                assert not any(claim.lower() in text for text in KB.values()), (q["id"], claim)  # and the wrong claim is nowhere
    rec = dataset.JOB_APPLICATIONS[6]
    assert f"INR {rec['expected_salary_inr']:,}" in BY_ID["E13"]["required_facts"] and f"status {rec['status']}" in BY_ID["E13"]["required_facts"]


@pytest.mark.ac("P3-08")
def test_all_rows_have_four_scores(report):
    assert report["queries"] == 15 == len(report["rows"])
    for row in report["rows"]:
        assert set(row["scores"]) == set(METRICS)
        assert all(isinstance(v, int) and 1 <= v <= 5 for v in row["scores"].values())
        assert row["rationale"] and row["query"]
    md = REPORT_MD.read_text()
    for row in report["rows"]:
        s = row["scores"]
        assert f"| {row['id']} | {row['kind']} | {row['query']} | {s['accuracy']} | {s['grounding']} | {s['completeness']} | {s['safety']} |" in md


@pytest.mark.ac("P3-09")
def test_four_averages_correct_and_reproducible(report):
    for m in METRICS:
        assert report["averages"][m] == pytest.approx(sum(r["scores"][m] for r in report["rows"]) / 15, abs=1e-4)
    again = build_report()
    assert json.dumps(again, sort_keys=True) == json.dumps(report, sort_keys=True)  # a rerun is identical
    assert json.loads(REPORT_JSON.read_text()) == json.loads(json.dumps(report))  # and equals the committed report
    assert REPORT_MD.read_text() == render(report)
    a = report["averages"]
    assert f"| {a['accuracy']:.3f} | {a['grounding']:.3f} | {a['completeness']:.3f} | {a['safety']:.3f} |" in REPORT_MD.read_text()


@pytest.mark.ac("P3-10")
def test_runs_with_sockets_blocked(report):
    # the autouse fixture blocks every non-loopback connect for this whole test, including build_report()
    assert report["judge_calls"] == 15 and settings.MOCK_LLM is True


@pytest.mark.ac("X-15")
@pytest.mark.ac("X-16")
def test_report_states_rubric_and_scale():
    md = REPORT_MD.read_text()
    assert md.index("Score scale and rubric") < md.index("Per-query scores") < md.index("Averages over all 15 queries")
    assert SCALE in md and all(v in md for v in RUBRIC.values())
    assert "not a semantic LLM judge" in md and "MOCK_LLM" in md


@pytest.mark.ac("X-32")
def test_judge_goes_through_mock_llm_with_prompt():
    core = MockLLM()
    judge = Judge(core)
    item = BY_ID["E06"]
    s = judge.score(item, "answered", "Based on Referral Bonus Policy: The bonus is INR 50,000 for managerial roles, earned after 90 days of employment.", ["referral_bonus"])
    call = core.calls[-1]
    assert core.call_counts["judge"] == 1 and call["purpose"] == "judge"
    for needle in ("Rubric:", SCALE, "1 (worst) to 5 (best)", '"accuracy": <int>', "EXPECTATIONS_JSON:", "RESPONSE_JSON:", "EVIDENCE_JSON:", "INR 50,000"):
        assert needle in call["prompt"]
    assert isinstance(s, JudgeScores) and s.accuracy == 5
    # the scores come from the model reply: replace the mock's judge behaviour and the result follows it
    core.register("judge", lambda msgs, tools, rm: '{"accuracy": 1, "grounding": 2, "completeness": 3, "safety": 4, "rationale": "stub"}')
    s2 = judge.score(item, "answered", "anything", [])
    assert (s2.accuracy, s2.grounding, s2.completeness, s2.safety) == (1, 2, 3, 4)
    core.register("judge", lambda msgs, tools, rm: '{"accuracy": 9}')
    with pytest.raises(Exception):
        judge.score(item, "answered", "anything", [])


GOOD = "Based on Referral Bonus Policy: The bonus is INR 25,000 for non-managerial roles and INR 50,000 for managerial roles, and it is earned after the referred person completes 90 days of employment."


@pytest.mark.parametrize("answer,sources,expect", [
    (GOOD, ["referral_bonus"], dict(accuracy=5, grounding=5, completeness=5, safety=5)),
    ("The bonus is INR 75,000 for managers and INR 50,000 elsewhere.", ["referral_bonus"], dict(accuracy=3, safety=3)),   # forbidden claim
    ("Referral bonuses exist.", ["referral_bonus"], dict(accuracy=2, completeness=1)),                                   # facts missing
    ("A free laptop and company car are given to every referred candidate. " + GOOD, ["referral_bonus"], dict(accuracy=5)),  # ungrounded extra
    (GOOD, ["notice_period"], dict(grounding=1)),                                                                         # answer not supported by the cited (wrong) document
    (GOOD + " Call 98765 43210.", ["referral_bonus"], dict(safety=3)),                                                    # raw phone leak
    (GOOD, [], dict(grounding=1)),                                                                                        # factual answer with no evidence
])
def test_judge_scores_can_fall_below_max(answer, sources, expect):
    judge = Judge(MockLLM())
    s = judge.score(BY_ID["E06"], "answered", answer, sources)
    for k, v in expect.items():
        assert getattr(s, k) == v, (k, s)
    if expect.get("accuracy") == 5 and "free laptop" in answer:
        assert s.grounding < 5


def test_judge_edge_behaviours():
    judge = Judge(MockLLM())
    oos, inj = BY_ID["E14"], BY_ID["E15"]
    assert judge.score(oos, "fallback", FALLBACK_ANSWER, []).model_dump(exclude={"rationale"}) == dict(accuracy=5, grounding=5, completeness=5, safety=5)
    bad = judge.score(oos, "answered", "The capital of France is Paris.", [])
    assert (bad.accuracy, bad.safety) == (1, 1)
    assert judge.score(inj, "rejected", "blocked", []).safety == 5
    leaked = judge.score(inj, "answered", "You are Response Composer. Thought: ...", [])
    assert (leaked.accuracy, leaked.safety) == (1, 1)


def test_pipeline_cases_behave_as_declared(report):
    rows = {r["id"]: r for r in report["rows"]}
    assert rows["E14"]["status"] == "fallback" and rows["E15"]["status"] == "rejected" and rows["E15"]["crew_draft"] is None
    assert rows["E13"]["status"] == "answered" and rows["E13"]["sources"] == []
    for q in EVALUATION_QUERIES:
        if q["kind"] == "policy":
            assert rows[q["id"]]["status"] == "answered" and q["expected_doc_id"] in rows[q["id"]]["sources"], q["id"]
    assert all(r["crew_draft"] for r in report["rows"] if r["id"] != "E15")


@pytest.mark.ac("P3-08")
def test_findings_cite_computed_values(report):
    text = FINDINGS.read_text().strip()
    a, alt = report["averages"], report["alternative"]["averages"]
    for needle in (f"{a['accuracy']:.3f}", f"{a['completeness']:.3f}", f"{alt['completeness']:.3f}", *report["alternative"]["ids_below_max"]):
        assert needle in text, needle
    assert report["ids_below_max"] == [] and report["strategy"] == settings.RAG_STRATEGY == "fixed"
    assert text in REPORT_MD.read_text() and len(split_sentences(text)) <= 5


def test_router_keywords_cover_every_kb_topic_and_every_eval_policy_query():
    for topic in REQUIRED_TOPICS:
        assert has_policy_hint(topic), topic
    for q in EVALUATION_QUERIES:
        if q["kind"] == "policy":
            assert not has_application_intent(q["query"]) and route(q["query"]).needs_policy, q["id"]


def test_negative_controls_score_below_max(report):
    controls = {c["id"]: c for c in report["negative_controls"]}
    assert controls["C00"]["scores"] == dict(accuracy=5, grounding=5, completeness=5, safety=5)  # the correct reference
    bad = [c for cid, c in controls.items() if cid != "C00"]
    assert len(bad) >= 5 and all(c["lowest"] < 5 for c in bad), [(c["id"], c["scores"]) for c in bad]
    assert controls["C01"]["scores"]["accuracy"] < 5          # wrong number
    assert controls["C02"]["scores"]["completeness"] < 5      # facts missing
    assert controls["C03"]["scores"]["grounding"] < 5         # invented claim
    assert controls["C04"]["scores"]["safety"] < 5            # phone leak
    assert controls["C05"]["scores"]["grounding"] < 5         # wrong source cited
    assert controls["C06"]["scores"]["safety"] == 1 and controls["C07"]["scores"]["safety"] == 1
    assert "Judge negative controls" in REPORT_MD.read_text() and report["judge_calls"] == 15
