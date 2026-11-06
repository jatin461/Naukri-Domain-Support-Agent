#!/usr/bin/env bash
# Regenerates every report and transcript, runs the whole test suite with a JUnit report, and builds the acceptance matrix.
# Prerequisites: the virtual environment (.venv) with the pinned requirements, and the embedding model in the local cache
# (python -m scripts.prefetch_model, the only step that needs the network). Everything below runs offline with MOCK_LLM and no API key.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export MOCK_LLM=true HF_HUB_OFFLINE=1
unset OPENAI_API_KEY || true

"$PY" -m scripts.prefetch_model --check
"$PY" dataset.py
"$PY" -m scripts.build_index
"$PY" -m scripts.calibrate_threshold
"$PY" -m scripts.compare_chunking
"$PY" -m scripts.escalation_report
"$PY" -m evaluation.run_evaluation
"$PY" -m scripts.generate_transcripts
mkdir -p .acceptance
"$PY" -m pytest -q --junitxml=.acceptance/junit.xml
"$PY" -m scripts.acceptance_report .acceptance/junit.xml
