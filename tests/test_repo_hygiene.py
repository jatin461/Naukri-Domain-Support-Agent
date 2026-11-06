import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROHIBITED = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg", ".ico", ".pdf", ".ppt", ".pptx", ".key", ".odp",
              ".mp4", ".mov", ".avi", ".mkv", ".webm", ".mp3", ".wav", ".flac", ".m4a", ".ogg", ".doc", ".docx", ".zip", ".bin", ".sqlite3"}
ALLOWED_TOP_LEVEL = {"README.md", "requirements.txt", "requirements.lock", ".python-version", ".env.example", ".gitignore", "pytest.ini",
                     "dataset.py", "data", "knowledge_base", "src", "evaluation", "scripts", "transcripts", "logs", "tests", "docs", "specs"}
REQUIRED_PATHS = [
    "README.md", "requirements.txt", ".env.example", "dataset.py", "data/generated_job_applications.json",
    *[f"knowledge_base/{n}.md" for n in ("job_application_eligibility", "interview_scheduling", "offer_negotiation", "background_verification",
                                          "notice_period", "referral_bonus", "internal_transfer", "probation_period", "remote_work",
                                          "diversity_hiring", "exit_interview", "applicant_data_retention")],
    "src/config/settings.py", "src/rag/chunking.py", "src/rag/embeddings.py", "src/rag/vector_store.py", "src/rag/retrieval.py",
    "src/rag/grounded_generation.py", "src/tools/rag_tool.py", "src/tools/application_lookup.py", "src/llm/mock_crewai_llm.py",
    "src/crew/agents.py", "src/crew/tasks.py", "src/crew/crew.py", "src/memory/session_memory.py", "src/guardrails/input_guardrails.py",
    "src/guardrails/output_guardrails.py", "src/schemas/crew_response.py", "src/review/autogen_review.py", "src/governance/autonomy.py",
    "src/governance/budget.py", "src/governance/risk_assessment.md", "src/cache/grounded_cache.py", "src/observability/logging_config.py",
    "src/api/main.py", "src/pipeline.py", "evaluation/evaluation_dataset.py", "evaluation/judge.py", "evaluation/reports",
    "transcripts/rag_demos.md", "transcripts/crew_tool_invocations.md", "transcripts/memory_sessions.md", "transcripts/guardrail_demos.md",
    "transcripts/autogen_review_demos.md", "transcripts/governance_budget_demo.md", "transcripts/cache_demo.md",
    *[f"tests/test_{n}.py" for n in ("dataset", "rag", "tools_and_crew", "memory", "guardrails", "api", "evaluation", "review", "governance_cache")],
    "logs/sample_requests.jsonl",
]
# Transcript or report per task 1-16 (traceability section 3b)
TASK_ARTEFACTS = {
    1: ["transcripts/dataset_distribution.txt"], 2: ["transcripts/kb_validation.txt"], 3: ["transcripts/rag_indexing.md"],
    4: ["transcripts/rag_demos.md", "evaluation/reports/similarity_calibration.md"], 5: ["evaluation/reports/chunking_comparison.md"],
    6: ["evaluation/reports/escalation_threshold.md", "transcripts/crew_tool_invocations.md"], 7: ["transcripts/crew_tool_invocations.md"],
    8: ["transcripts/memory_sessions.md"], 9: ["transcripts/structured_output.md"], 10: ["transcripts/guardrail_demos.md"],
    11: ["transcripts/api_demo.md"], 12: ["logs/sample_requests.jsonl", "transcripts/logging_privacy_demo.md"],
    13: ["evaluation/reports/15_query_results.md"], 14: ["transcripts/autogen_review_demos.md"],
    15: ["transcripts/governance_autonomy_demo.md", "src/governance/risk_assessment.md", "transcripts/governance_budget_demo.md"],
    16: ["transcripts/cache_demo.md"],
}


def repo_files():
    """Files git would track (tracked or untracked, minus ignored); falls back to a filtered walk."""
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return [ROOT / p for p in out.splitlines() if (ROOT / p).is_file()]
    except Exception:  # pragma: no cover
        skip = {".venv", ".git", "chroma_db", "__pycache__", ".pytest_cache", ".acceptance"}
        return [p for p in ROOT.rglob("*") if p.is_file() and not skip & set(p.relative_to(ROOT).parts)]


@pytest.mark.ac("S-07")
def test_no_prohibited_file_types():
    files = repo_files()
    assert files
    bad = [str(p.relative_to(ROOT)) for p in files if p.suffix.lower() in PROHIBITED or p.name == ".DS_Store"]
    assert not bad, bad
    for p in files:  # code and text only: everything decodes as UTF-8 and is small
        p.read_text(encoding="utf-8")
        assert p.stat().st_size < 1_000_000, p


def test_only_required_top_level_entries_are_tracked():
    top = {p.relative_to(ROOT).parts[0] for p in repo_files()}
    assert top <= ALLOWED_TOP_LEVEL, sorted(top - ALLOWED_TOP_LEVEL)
    names = {p.name for p in repo_files()}
    assert not {"__pycache__", ".pytest_cache", "chroma_db"} & {part for p in repo_files() for part in p.relative_to(ROOT).parts}
    assert ".DS_Store" not in names


def test_required_paths_from_the_requirements_layout_exist():
    missing = [p for p in REQUIRED_PATHS if not (ROOT / p).exists()]
    assert not missing, missing


@pytest.mark.ac("X-19")
def test_no_secrets_in_env_example_or_logs():
    env = (ROOT / ".env.example").read_text()
    keys = re.findall(r"^([A-Z_]+)=(.*)$", env, re.MULTILINE)
    assert keys and not any(re.search(r"KEY|SECRET|TOKEN|PASSWORD", k) for k, _ in keys)
    assert not re.search(r"sk-[A-Za-z0-9]{8,}|AKIA[0-9A-Z]{12,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}", "\n".join(p.read_text() for p in repo_files() if p.suffix in {".py", ".md", ".txt", ".jsonl", ".ini", ".example", ""} and p.name != "requirements.lock"))
    logs = (ROOT / "logs" / "sample_requests.jsonl").read_text()
    assert not re.search(r"api[_-]?key|secret|password|bearer|authorization|openai", logs, re.IGNORECASE)


@pytest.mark.ac("X-21")
def test_all_required_transcripts_exist_and_nonempty():
    names = ["dataset_distribution.txt", "kb_validation.txt", "rag_indexing.md", "rag_demos.md", "crew_tool_invocations.md", "memory_sessions.md",
             "structured_output.md", "guardrail_demos.md", "api_demo.md", "logging_privacy_demo.md", "autogen_review_demos.md",
             "governance_autonomy_demo.md", "governance_budget_demo.md", "cache_demo.md"]
    for n in names:
        assert (ROOT / "transcripts" / n).read_text().strip(), n
    for n in ("similarity_calibration", "chunking_comparison", "escalation_threshold", "15_query_results"):
        assert (ROOT / "evaluation" / "reports" / f"{n}.md").read_text().strip() and (ROOT / "evaluation" / "reports" / f"{n}.json").exists(), n
    assert len((ROOT / "logs" / "sample_requests.jsonl").read_text().splitlines()) >= 10


@pytest.mark.ac("X-31")
def test_transcript_for_every_task():
    assert sorted(TASK_ARTEFACTS) == list(range(1, 17))
    for task, paths in TASK_ARTEFACTS.items():
        for p in paths:
            assert (ROOT / p).read_text().strip(), (task, p)


@pytest.mark.ac("X-25")
def test_phone_fixtures_are_fabricated():
    pattern = re.compile(r"(?<!\d)(?:\+91[\s-]?)?\d{5}[\s-]?\d{5}(?!\d)")
    fake_digits = "9876543210"  # the only fabricated number used in the repository (98765 43210 in several formats)
    found = 0
    for p in repo_files():
        if p.name in {"requirements.lock"} or p.suffix not in {".py", ".md", ".txt", ".jsonl", ".json", ".ini", ".example", ""}:
            continue
        for m in pattern.finditer(p.read_text(encoding="utf-8")):
            digits = re.sub(r"\D", "", m.group(0))
            digits = digits[-10:]
            assert digits == fake_digits, (str(p.relative_to(ROOT)), m.group(0))
            found += 1
    assert found > 0
