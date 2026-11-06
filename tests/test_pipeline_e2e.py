import json

import pytest
from fastapi.testclient import TestClient

import dataset
from src.api.main import create_app
from src.pipeline import Pipeline
from src.schemas.crew_response import CrewResponse

APP3 = dataset.JOB_APPLICATIONS[2]


@pytest.mark.ac("X-27")
def test_full_request_touches_every_block(production_collection, core, tmp_path):
    log = tmp_path / "log.jsonl"
    pipe = Pipeline(collection=production_collection, llm_core=core)
    client = TestClient(create_app(pipe, log))
    q = "Contact: +91-98765-43210. What is the status of APP-0003 and the probation policy?"
    r = client.post("/ask", json={"question": q, "session_id": "e2e"}).json()
    assert r["status"] == "answered"
    # mask -> inject-check -> budget -> memory -> crew with both tools -> schema -> groundedness -> AutoGen -> log
    entry = json.loads(log.read_text().splitlines()[0])
    assert entry["phone_numbers_masked"] == 1 and "98765" not in log.read_text() and "[PHONE_MASKED]" in entry["request_text"]
    assert entry["tools"] == "rag_lookup,check_job_application_status" and entry["sources"].startswith("probation_period")
    assert entry["budget_allowed"] is True and entry["groundedness_action"] == "pass" and entry["review_approved"] is True
    assert entry["cache_misses"] == 1 and entry["http_status"] == 200 and entry["duration_ms"] > 0
    resp = CrewResponse.model_validate(r["response"])
    assert resp.response_type == "policy_and_lookup" and resp.record_id == "APP-0003" and resp.expected_salary_inr == APP3["expected_salary_inr"]
    assert pipe.sessions.get_history("e2e").messages[0].content.startswith("Contact: [PHONE_MASKED]")  # memory holds masked text
    prompts = "\n".join(c["prompt"] for c in core.calls)
    assert "98765" not in prompts and core.call_counts["review"] == 2 and core.call_counts["crew"] >= 3
    follow = client.post("/ask", json={"question": "and what is its expected salary?", "session_id": "e2e"}).json()
    assert follow["response"]["record_id"] == "APP-0003"  # a later turn uses the earlier one
