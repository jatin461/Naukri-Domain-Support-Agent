import os
import socket
import subprocess
import sys

import pytest

from src.config import settings

FLAGS = {
    "CREWAI_DISABLE_TELEMETRY": "true",
    "OTEL_SDK_DISABLED": "true",
    "CREWAI_DISABLE_TRACKING": "true",
    "ANONYMIZED_TELEMETRY": "False",
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


@pytest.mark.ac("P2-11")
def test_telemetry_env_set_before_crewai_import():
    code = (
        "import os, sys\n"
        "for k in %r: os.environ.pop(k, None)\n"
        "assert 'crewai' not in sys.modules\n"
        "import src.config.settings\n"
        "assert 'crewai' not in sys.modules and 'chromadb' not in sys.modules and 'sentence_transformers' not in sys.modules\n"
        "assert os.environ['CREWAI_DISABLE_TELEMETRY'] == 'true' and os.environ['OTEL_SDK_DISABLED'] == 'true'\n"
        "import crewai\n"
        "assert os.environ['CREWAI_DISABLE_TELEMETRY'] == 'true'\n"
        "print('ok')\n" % (list(FLAGS) + ["MOCK_LLM"],)
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=settings.PROJECT_ROOT, env={**os.environ, "PYTHONPATH": "."},
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-400:]


@pytest.mark.ac("X-12")
def test_all_telemetry_flags_off():
    for k, v in FLAGS.items():
        assert os.environ[k] == v, k
    assert "OPENAI_API_KEY" not in os.environ


@pytest.mark.ac("X-11")
def test_outbound_socket_raises():
    s = socket.socket()
    try:
        with pytest.raises(RuntimeError, match="network blocked"):
            s.connect(("93.184.216.34", 80))
    finally:
        s.close()


@pytest.mark.ac("X-22")
def test_mock_llm_false_is_not_implemented():
    code = "import os; os.environ['MOCK_LLM']='false'; import src.config.settings"
    r = subprocess.run([sys.executable, "-c", code], cwd=settings.PROJECT_ROOT, env={**os.environ, "PYTHONPATH": "."},
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "NotImplementedError" in r.stderr


@pytest.mark.ac("X-23")
def test_all_four_consumers_run_mock_with_sockets_blocked(production_collection):
    from evaluation.evaluation_dataset import EVALUATION_QUERIES
    from evaluation.judge import Judge
    from src.llm.mock_llm import MockLLM
    from src.pipeline import Pipeline

    core = MockLLM()  # one shared deterministic core behind every consumer
    pipe = Pipeline(collection=production_collection, llm_core=core)
    r = pipe.handle("How much referral bonus is paid for a managerial role?", "four")  # grounded generation, CrewAI crew, AutoGen review
    Judge(core).score(EVALUATION_QUERIES[5], r.status, r.answer, r.response.sources)  # evaluation judge
    assert all(core.call_counts[p] > 0 for p in ("grounded_answer", "crew", "review", "judge")), dict(core.call_counts)
    assert r.review is not None and r.status == "answered"
