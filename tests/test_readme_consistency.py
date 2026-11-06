import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import pytest

import dataset
from src.config import settings
from src.tools.application_lookup import ESCALATION_THRESHOLD

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
REPORTS = settings.REPORTS_DIR
HW = ROOT / "evaluation" / "hand_written"


def section(n):
    m = re.search(rf"^## {n}\. .*?$(.*?)(?=^## \d+\. |\Z)", README, re.MULTILINE | re.DOTALL)
    assert m, f"README section {n} missing"
    return m.group(1)


def js(name):
    return json.loads((REPORTS / name).read_text())


@pytest.mark.ac("S-02")
def test_track_stated_first():
    lines = [l for l in README.splitlines() if l.strip()]
    assert lines[0].startswith("# ") and "Naukri.com" in lines[0]
    assert "Naukri.com (Recruitment & HR)" in lines[1] and "Completed track" in lines[1]
    assert README.index("Naukri.com (Recruitment & HR)") < README.index("## 1.")


@pytest.mark.ac("S-03")
@pytest.mark.ac("P1-05")
def test_dataset_choices_match_code():
    s = section(1)
    assert f"**Random seed:** {dataset.SEED}" in s
    for k, v in dataset.CATEGORY_WEIGHTS.items():
        assert f"{k} {v:.2f}" in s
    for k, v in dataset.STATUS_WEIGHTS.items():
        assert f"{k} {v:.2f}" in s
    assert f"**Priority-flag probability:** {dataset.FLAG_PROBABILITY}" in s
    assert f"INR {dataset.SALARY_MIN_INR:,} to INR {dataset.SALARY_MAX_INR:,}" in s
    assert dataset.SALARY_RATIONALE in s  # the one-sentence salary reasoning
    assert f"**Records:** {dataset.N_RECORDS}" in s
    cat = Counter(r["category"] for r in dataset.JOB_APPLICATIONS)
    st = Counter(r["status"] for r in dataset.JOB_APPLICATIONS)
    for k, n in {**cat, **st}.items():
        assert re.search(rf"\| {re.escape(k)} \| {n} \|", s), k  # printed counts per category and status
    flag = sum(r["flagged_priority_review"] for r in dataset.JOB_APPLICATIONS)
    assert f"{flag}/{dataset.N_RECORDS} = {100 * flag / dataset.N_RECORDS:.2f}%" in s
    # and the README stands at the top: dataset section precedes everything that is not the intro
    assert README.index("## 1.") < README.index("## 4.")


@pytest.mark.ac("S-04")
@pytest.mark.ac("X-22")
def test_readme_confirms_offline():
    s = section(2)
    for needle in ("`MOCK_LLM`", "zero API keys", "no network", "scripts.prefetch_model", "HF_HUB_OFFLINE=1", "socket.connect",
                   "MOCK_LLM=false", "NotImplementedError", "grounded generation", "evaluation judge", "CrewAI crew", "AutoGen review team"):
        assert needle.lower() in s.lower(), needle
    assert settings.MOCK_LLM is True and "OPENAI_API_KEY" not in (ROOT / ".env.example").read_text()


@pytest.mark.ac("S-05")
@pytest.mark.ac("P2-11")
def test_readme_confirms_telemetry():
    s = section(3)
    assert "CREWAI_DISABLE_TELEMETRY=true" in s and "OTEL_SDK_DISABLED=true" in s
    assert "before" in s and "`crewai`" in s and ".kickoff()" in s
    import os

    assert os.environ["CREWAI_DISABLE_TELEMETRY"] == "true" and os.environ["OTEL_SDK_DISABLED"] == "true"


@pytest.mark.ac("S-06")
def test_readme_similarity_matches_report():
    s = section(4)
    rep = js("similarity_calibration.json")
    for strat in ("fixed", "sentence"):
        r = rep[strat]
        for x in r["in_scope"] + r["out_of_scope"]:
            assert f"| {strat} |" in s and f"{x['top1_similarity']:.4f}" in s and x["query"] in s
        assert f"{r['threshold']:.3f}" in s and f"{r['gap']:.4f}" in s
    chosen = rep[settings.RAG_STRATEGY]["threshold"]
    assert f"**Selected fallback threshold: {chosen:.3f}**" in s
    assert settings.get_similarity_threshold() == chosen
    assert (HW / "threshold_rationale.md").read_text().strip() in s


def test_readme_chunking_section_matches_report():
    s = section(5)
    rep = js("chunking_comparison.json")
    for strat in ("fixed", "sentence"):
        a = rep[strat]["aggregate"]
        assert f"{a['mean_precision']:.3f} / {a['mean_recall']:.3f}" in s
        for r in rep[strat]["rows"]:
            assert f"{r['hits']}/{r['n_retrieved']} = {r['precision']:.3f}; {r['hits']}/{r['n_relevant']} = {r['recall']:.3f}" in s
    assert (HW / "chunking_recommendation.md").read_text().strip() in s


@pytest.mark.ac("X-01")
def test_readme_escalation_formula():
    s = section(6)
    rep = js("escalation_threshold.json")
    assert f"`escalation_score = {rep['formula']}`" in s
    assert f"**Escalation is recommended when the score is at least {ESCALATION_THRESHOLD}.**" in s
    p = rep["days_percentiles"]
    for needle in (f"P20 = {p['p20']:.2f}", f"P80 = {p['p80']:.2f}", f"{rep['escalated']} of {rep['records']}"):
        assert needle in s, needle
    assert (HW / "escalation_justification.md").read_text().strip() in s


@pytest.mark.ac("X-13")
@pytest.mark.ac("X-14")
def test_readme_states_masking_scope_and_injection_limits():
    s = section(7)
    assert "Only the fixed-format phone number" in s
    assert "not masked" in s and "Candidate names, expected salary and background-check results" in s
    assert not re.search(r"(names?|salary|background[- ]check)[^.]{0,40}(are|is) (reliably )?masked", s.replace("are not masked", "").replace("is not masked", ""), re.IGNORECASE)
    for needle in ("Limitations", "paraphrases", "other languages", "obfuscated", "not a complete detector"):
        assert needle in s, needle
    assert "`[PHONE_MASKED]`" in s and "mask_phone" in s


def test_readme_governance_scope_and_cache_claims():
    s = section(8)
    for layer in ("Data layer", "Model layer", "Application layer", "Runtime layer"):
        assert f"**{layer}" in s, layer
    assert "**Risk classification: Medium.**" in s and len(re.findall(r"\*\*Risk classification: (Low|Medium|High)\.", s)) == 1
    assert "does not match paraphrases" in s and "no semantic-match claim" in s
    assert "nothing is charged" in s and "AutonomyViolation" in s


def test_readme_evaluation_matches_report():
    s = section(9)
    rep = js("15_query_results.json")
    a = rep["averages"]
    assert f"| {a['accuracy']:.3f} | {a['grounding']:.3f} | {a['completeness']:.3f} | {a['safety']:.3f} |" in s
    assert f"completeness {rep['alternative']['averages']['completeness']:.3f}" in s
    assert "not a semantic LLM judge" in s and (HW / "evaluation_findings.md").read_text().strip() in s


@pytest.mark.ac("X-02")
def test_readme_commands_reference_existing_files_and_modules():
    cmds = re.findall(r"```bash\n(.*?)\n```", README, re.DOTALL)
    assert len(cmds) >= 7
    for block in cmds:
        for line in block.splitlines():
            for mod in re.findall(r"python -m ([A-Za-z0-9_.]+)", line):
                if mod in ("pytest", "uvicorn", "venv"):
                    continue
                assert (ROOT / (mod.replace(".", "/") + ".py")).exists(), (mod, line)
            for path in re.findall(r"(?:bash|python) ((?:scripts/|src/)[A-Za-z0-9_/.]+)", line):
                assert (ROOT / path).exists(), path
    assert (ROOT / "scripts" / "run_acceptance.sh").exists() and "src.api.main:app" in README
    from src.api.main import app  # the documented uvicorn target imports without a built index

    assert {"/ask", "/add-document", "/chat"} <= {getattr(r, "path", None) for r in app.routes}


def test_readme_originality_statement_and_layout_lists_real_paths():
    s = section(14)
    assert "written for this project" in s and "8-word run" in s and "fabricated" in s
    layout = re.search(r"```text\n(.*?)\n```", section(12), re.DOTALL).group(1)
    for token in re.findall(r"^(?:\S+/|\S+\.py)", layout, re.MULTILINE):
        assert (ROOT / token.rstrip("/")).exists() or token.startswith("README"), token


def test_readme_uses_no_images_and_makes_no_previous_version_references():
    assert not re.search(r"!\[[^\]]*\]\(", README)
    assert not re.search(r"previous (version|revision)|earlier (version|revision)|originally (said|stated)|superseded", README, re.IGNORECASE)
