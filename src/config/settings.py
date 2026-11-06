"""Offline / telemetry environment. Import this BEFORE crewai, chromadb or sentence_transformers."""
import os

_OFFLINE_ENV = {
    "MOCK_LLM": "true",
    "CREWAI_DISABLE_TELEMETRY": "true",
    "OTEL_SDK_DISABLED": "true",
    "CREWAI_DISABLE_TRACKING": "true",
    "ANONYMIZED_TELEMETRY": "False",
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
for _k, _v in _OFFLINE_ENV.items():
    os.environ.setdefault(_k, _v)
os.environ.pop("OPENAI_API_KEY", None)

MOCK_LLM = os.environ["MOCK_LLM"].lower() == "true"
if not MOCK_LLM:
    raise NotImplementedError("Only the deterministic MOCK_LLM mode is implemented.")

# --- RAG configuration (Phase 2) ---
from pathlib import Path as _Path  # noqa: E402

PROJECT_ROOT = _Path(__file__).resolve().parents[2]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = _Path(os.environ.get("CHROMA_DIR", PROJECT_ROOT / "chroma_db"))
COLLECTION_NAMES = {"fixed": "kb_fixed_overlap", "sentence": "kb_sentence"}
TOP_K = 3
FIXED_CHUNK_WORDS = 30
FIXED_CHUNK_OVERLAP_WORDS = 8
REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"

# Strategy chosen by the Task 5 comparison (see evaluation/reports/chunking_comparison.md).
RAG_STRATEGY = "fixed"


def get_similarity_threshold(strategy=None):
    """Fallback threshold measured by scripts/calibrate_threshold.py for the chosen collection."""
    import json

    report = json.loads((REPORTS_DIR / "similarity_calibration.json").read_text(encoding="utf-8"))
    return report[strategy or RAG_STRATEGY]["threshold"]

MAX_REQUEST_CHARS = 20_000  # hard validation cap; the (lower) token budget is enforced separately in Phase 8
GROUNDING_SUPPORT_THRESHOLD = 0.6  # fraction of a sentence's content words that must appear in the evidence

LOG_PATH = _Path(os.environ.get("REQUEST_LOG_PATH", PROJECT_ROOT / "logs" / "requests.jsonl"))
MAX_DOCUMENT_CHARS = 5_000  # /add-document size cap (design choice; the 12 required documents are about 400 characters each)

# Per-request budget (simulated; see src/governance/budget.py). No provider is called in MOCK_LLM mode, so nothing is charged.
MAX_REQUEST_TOKENS = 1_000
PROMPT_OVERHEAD_TOKENS = 600  # allowance for agent prompts, retrieved context and the review prompt
SIMULATED_USD_PER_1K_TOKENS = 0.002  # illustrative rate used only to express the estimate as a cost
