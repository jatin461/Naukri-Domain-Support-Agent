# Requirements Traceability Matrix

**Project:** Naukri.com Domain Support Agent (Recruitment & HR) — CrewAI + AutoGen + RAG + FastAPI
**Authority:** `specs/naukri_capstone_requirements.md`.
**Document status:** every criterion below is covered by an executed test and an evidence artefact, except S-01, which needs the repository owner to publish the repository. The authoritative pass/fail matrix is generated from executed tests by `scripts/acceptance_report.py` into `evaluation/reports/acceptance_matrix.md`.
**Authority order:** the capstone brief ("Final Capstone — Naukri.com Domain Support Agent (CrewAI)"), then the requirements spec, then the architecture and implementation plan. §2b maps every clause of the brief to criteria; §3 tracks the X-items; §3b maps every one of the 16 tasks to a transcript, because the brief requires "transcripts demonstrating every task above".
**Conventions**

- File paths and test names are the ones in this repository.
- `AC` = acceptance criterion from Requirements §10, numbered in checklist order. `X` = requirement that appears in the requirements text (§1, §5–§8) but **not** in the §10 checklist; these are easy to miss and are tracked separately in §3.
- Verification commands assume the project venv is active and `MOCK_LLM=true`, with the offline env vars from `docs/dependency_compatibility.md`.
- A criterion is only `Done` when its verification command passes **and** its evidence artefact exists in the repo. Evidence artefacts are produced by scripts (`scripts/generate_transcripts.py`), never hand-written, so transcripts cannot drift from behaviour.
- Marks are specified per Part (30/30/20/20), not per task or per criterion.

---

## 1. The 16 tasks

| # | Part | Task | Spec § | Criteria |
|---|---|---|---|---|
| 1 | 1 (30) | Deterministic job-application dataset | 4.1 | P1-01…P1-05 |
| 2 | 1 | Knowledge-base authoring (12 topics) | 4.2, 3.3 | P1-06, P1-07 |
| 3 | 1 | Two chunking strategies, two ChromaDB collections, `upsert()` | 4.3 | P1-08…P1-11 |
| 4 | 1 | Grounded generation + empirical fallback calibration | 4.4 | P1-12…P1-14 |
| 5 | 1 | Document-level precision/recall chunking comparison | 4.5 | P1-15…P1-17 |
| 6 | 2 (30) | `check_job_application_status` + escalation score | 5.1 | P2-01…P2-03 |
| 7 | 2 | CrewAI crew (3 agents), `.kickoff()`, `BaseLLM` mock | 5.2, 5.3 | P2-04…P2-11 |
| 8 | 2 | In-process session memory (LangChain) | 5.4 | P2-12, P2-13 |
| 9 | 2 | Structured (Pydantic) crew response | 5.5 | P2-14 |
| 10 | 2 | Guardrails: phone masking, injection, groundedness | 5.6 | P2-15…P2-17 |
| 11 | 3 (20) | FastAPI: 2 HTTP + 1 WebSocket, disconnect handling, concurrent-session isolation | 6.1 | P3-01…P3-04, P3-11 |
| 12 | 3 | JSONL logging with trace ID/timing, PII-safe | 6.2 | P3-05, P3-06 |
| 13 | 3 | 15-query evaluation, 4 scores/query + 4 averages | 6.3 | P3-07…P3-10 |
| 14 | 4 (20) | AutoGen 2-agent RoundRobin review, structured verdict | 7.1 | P4-01…P4-08 |
| 15 | 4 | Governance: least autonomy, risk class, per-request budget | 7.2 | P4-09…P4-13 |
| 16 | 4 | Grounded-generation response cache, normalization rule | 7.3 | P4-14…P4-16 |

Submission/README requirements (§1.2, §10 "Submission") → S-01…S-07.

> **Task 15 scope.** The brief titles it "Apply the four-layer AI-governance model" but itself specifies tasks for only **two** layers ("At the Application layer…", "At the Runtime layer…"). The other two layers are not named anywhere in the brief: implement and test exactly those two, and do not invent others. The README governance section will say that the brief's tasks address the Application and Runtime layers.

---

## 2. Acceptance-criteria matrix (Requirements §10)

Verification commands use pytest node IDs; `S:` prefix = a script whose output is the evidence.

### Part 1 — Dataset Design and RAG Core (30 marks)

| ID | Criterion | Task | Implementation | Test file | Verification command | Evidence artefact | Status |
|---|---|---|---|---|---|---|---|
| P1-01 | `dataset.py` deterministically generates ≥40 records | 1 | `dataset.py` | `tests/test_dataset.py` | `pytest tests/test_dataset.py::test_min_records_and_determinism` (two generations byte-identical) | `transcripts/dataset_distribution.txt` | Done |
| P1-02 | Every required category ≥3 times (exact 5 strings) | 1 | `dataset.py` | `tests/test_dataset.py` | `pytest tests/test_dataset.py::test_category_coverage` | same | Done |
| P1-03 | Every required status ≥1 time (exact 5 strings) | 1 | `dataset.py` | `tests/test_dataset.py` | `pytest tests/test_dataset.py::test_status_coverage` | same | Done |
| P1-04 | Priority-flag % in [10,30]; if first draw misses, change seed or weights and regenerate; never hand-edit records. Generation is a plain seeded **weighted draw** (documented category/status/priority weights), validated afterwards, as the brief literally describes — not a quota-forcing scheme | 1 | `dataset.py` (weighted generation; seed/weights adjusted, never records) | `tests/test_dataset.py` | `pytest tests/test_dataset.py::test_priority_percentage tests/test_dataset.py::test_snapshot_equals_regeneration` | same | Done |
| P1-05 | Record fields; one-sentence salary-range reasoning; README **top** states seed, **category weights, status weights** (and priority probability) and the **amount (salary) range**, so the grader can reproduce the dataset; counts per category/status and flagged % printed *and* reported | 1 | `dataset.py`, `README.md` | `tests/test_dataset.py`, `tests/test_readme_consistency.py` | `pytest tests/test_readme_consistency.py::test_dataset_choices_match_code` (README seed/weights/salary range parsed and compared to `dataset.py` constants); also `python dataset.py` prints category/status counts + priority % | `transcripts/dataset_distribution.txt` | Done |
| P1-06 | ≥12 original KB docs, each 2–5 sentences | 2 | `knowledge_base/*.md`, `src/rag/kb_loader.py` | `tests/test_kb.py` | `pytest tests/test_kb.py::test_doc_count_and_sentence_range` | `transcripts/kb_validation.txt` | Done |
| P1-07 | All 12 exact policy topics present (stable doc IDs) | 2 | `knowledge_base/*.md`, `kb_loader.py` (topic→doc-ID map) | `tests/test_kb.py` | `pytest tests/test_kb.py::test_all_twelve_topics_present` | same | Done |
| P1-08 | Fixed-size-with-overlap chunking (non-zero overlap, parent metadata, chunk IDs) | 3 | `src/rag/chunking.py` | `tests/test_rag.py` | `pytest tests/test_rag.py -k fixed_chunk` (short/long/empty/one-sentence docs; overlap > 0) | `transcripts/rag_demos.md` | Done |
| P1-09 | Sentence-based chunking (parent metadata, chunk IDs) | 3 | `src/rag/chunking.py` | `tests/test_rag.py` | `pytest tests/test_rag.py -k sentence_chunk` | same | Done |
| P1-10 | Both chunk sets embedded with local SentenceTransformers | 3 | `src/rag/embeddings.py` | `tests/test_rag.py` | `pytest tests/test_rag.py::test_embeddings_local_offline` (runs with `HF_HUB_OFFLINE=1` and sockets blocked) | same | Done |
| P1-11 | One ChromaDB collection per strategy, indexed via `collection.upsert()` | 3 | `src/rag/vector_store.py`, `scripts/build_index.py` | `tests/test_rag.py` | `pytest tests/test_rag.py::test_two_collections_use_upsert` (spy on `upsert`; names differ; re-index leaves `count()` unchanged; both return sensible top-1 for one query each) | `transcripts/rag_demos.md` | Done |
| P1-12 | Threshold empirically calibrated: ≥3 in-scope + ≥2 out-of-scope top-1 cosine values; chosen between clusters; not 0.5/0.6/0.7 by default | 4 | `src/rag/calibration.py`, `scripts/calibrate_threshold.py`, `src/config/settings.py` (reads chosen value from report) | `tests/test_rag.py` | `S: python -m scripts.calibrate_threshold`; `pytest tests/test_rag.py::test_threshold_between_clusters` (min(in-scope) > T > max(out-of-scope), values read from the committed report, README values match) | `evaluation/reports/similarity_calibration.md` | Done |
| P1-13 | ≥5 in-scope queries return context-grounded answers | 4 | `src/rag/grounded_generation.py`, `src/rag/retrieval.py` | `tests/test_rag.py` | `pytest tests/test_rag.py::test_five_in_scope_grounded` (answer sentences ⊆ retrieved text; source doc IDs returned) | `transcripts/rag_demos.md` | Done |
| P1-14 | ≥1 deliberately out-of-scope query triggers the fallback | 4 | `grounded_generation.py` | `tests/test_rag.py` | `pytest tests/test_rag.py::test_out_of_scope_fallback` | same | Done |
| P1-15 | Doc-level precision & recall for both collections on the same ≥5 queries (relevant docs declared before scoring; parent IDs deduplicated) | 5 | `src/rag/retrieval_eval.py`, `scripts/compare_chunking.py` | `tests/test_rag.py` | `S: python -m scripts.compare_chunking`; `pytest tests/test_rag.py::test_precision_recall_math` (hand-computed fixture) | `evaluation/reports/chunking_comparison.md` | Done |
| P1-16 | Per-query arithmetic shown for both strategies (not aggregates only) | 5 | `retrieval_eval.py` | `tests/test_rag.py` | `pytest tests/test_rag.py::test_report_shows_per_query_arithmetic` (report has ≥5 rows × 2 strategies, each with `hits/retrieved` and `hits/relevant`) | same | Done |
| P1-17 | 2–3 sentence recommendation citing actual numbers from both strategies (brief: "your own two sets of numbers"; originality applies, so the sentences are authored by hand after the run, and the test only verifies they cite the computed values) | 5 | `evaluation/reports/chunking_comparison.md` (hand-written recommendation block), `retrieval_eval.py` | `tests/test_rag.py` | `pytest tests/test_rag.py::test_recommendation_cites_numbers` (2–3 sentences; contains both aggregates) | same | Done |

### Part 2 — CrewAI orchestration, tools, memory, guardrails (30 marks)

| ID | Criterion | Task | Implementation | Test file | Verification command | Evidence artefact | Status |
|---|---|---|---|---|---|---|---|
| P2-01 | `check_job_application_status(record_id: str) -> dict` returns status, expected salary, `escalation_score`∈[0,1]; unknown ID handled | 6 | `src/tools/application_lookup.py` | `tests/test_escalation.py` | `pytest tests/test_escalation.py::test_known_unknown_and_bounds` (all dataset records ∈[0,1]; unknown → controlled not-found, no invented record) | `transcripts/crew_tool_invocations.md` | Done |
| P2-02 | Score genuinely combines priority flag and normalized recency (not boolean OR) | 6 | `application_lookup.py` (`compute_escalation_score`) | `tests/test_escalation.py` | `pytest tests/test_escalation.py::test_both_components_change_outcome` (flipping flag *and* varying days each flip the escalate decision for some record pair at the chosen threshold) | same | Done |
| P2-03 | Exact formula + threshold documented; threshold justified by percentile *computed* from the generated dataset | 6 | `application_lookup.py`, `scripts/escalation_report.py`, `README.md` | `tests/test_escalation.py`, `tests/test_readme_consistency.py` | `S: python -m scripts.escalation_report`; `pytest tests/test_escalation.py::test_percentile_computed_not_hardcoded` | `evaluation/reports/escalation_threshold.md` | Done |
| P2-04 | Crew has ≥3 agents: Retrieval, Lookup, Composer | 7 | `src/crew/agents.py`, `crew.py` | `tests/test_tools_and_crew.py` | `pytest tests/test_tools_and_crew.py::test_three_distinct_agents` | `transcripts/crew_tool_invocations.md` | Done |
| P2-05 | Crew runs via `.kickoff()` | 7 | `src/crew/crew.py`, `tasks.py` | `tests/test_tools_and_crew.py` | `pytest tests/test_tools_and_crew.py::test_kickoff_returns_output` | same | Done |
| P2-06 | RAG tool actually invoked (policy query) | 7 | `src/tools/rag_tool.py` | `tests/test_tools_and_crew.py` | `pytest tests/test_tools_and_crew.py::test_rag_tool_invoked_on_policy_query` (call counter / recorded invocation with args) | same | Done |
| P2-07 | Lookup tool actually invoked on a *different* query | 7 | `application_lookup.py` | `tests/test_tools_and_crew.py` | `pytest tests/test_tools_and_crew.py::test_lookup_tool_invoked_on_status_query` (RAG tool *not* invoked in this run) | same | Done |
| P2-08 | Mock LLM extends `crewai.llms.base_llm.BaseLLM` | 7 | `src/llm/mock_crewai_llm.py` | `tests/test_mock_llm.py` | `pytest tests/test_mock_llm.py::test_is_basellm_subclass` | — | Done |
| P2-09 | Parses model-generated text, not system prompt (`Observation:` pitfall) | 7 | `mock_crewai_llm.py` | `tests/test_mock_llm.py` | `pytest tests/test_mock_llm.py::test_first_turn_ignores_template_observation tests/test_mock_llm.py::test_final_answer_after_real_observation` | `transcripts/crew_tool_invocations.md` | Done |
| P2-10 | Tool args chosen from declared arg schema, not name substring (`rag_lookup` case) | 7 | `mock_crewai_llm.py` | `tests/test_mock_llm.py` | `pytest tests/test_mock_llm.py::test_rag_lookup_named_tool_not_misrouted tests/test_mock_llm.py::test_args_follow_declared_schema` | same | Done |
| P2-11 | Telemetry disabled **before** `.kickoff()` and confirmed in README | 7 | `src/config/settings.py` (sets env on import, before `crewai` import) | `tests/test_offline.py`, `tests/test_readme_consistency.py` | `pytest tests/test_offline.py::test_telemetry_env_set_before_crewai_import tests/test_readme_consistency.py::test_readme_confirms_telemetry` | — | Done |
| P2-12 | Multi-turn in-process memory demonstrated (earlier-turn context used later) | 8 | `src/memory/session_memory.py` | `tests/test_memory.py` | `pytest tests/test_memory.py::test_followup_uses_prior_turn` | `transcripts/memory_sessions.md` | Done |
| P2-13 | Separate fresh session shows reset/absence of prior state | 8 | `session_memory.py` | `tests/test_memory.py` | `pytest tests/test_memory.py::test_fresh_session_has_no_prior_state` | same | Done |
| P2-14 | Pydantic `BaseModel` that every crew response must conform to, supplied as the `response_format` (installed CrewAI exposes `Task.output_pydantic`/`Task.response_model` and an LLM `response_format` field — spike S2 picks the working one), **and** each response validated in code | 9 | `src/schemas/crew_response.py`, `crew.py` | `tests/test_tools_and_crew.py` | `pytest tests/test_tools_and_crew.py::test_every_crew_response_validated tests/test_tools_and_crew.py::test_malformed_output_fails_safely` | `transcripts/crew_tool_invocations.md` | Done |
| P2-15 | Fixed-format phone masking demonstrated before model input | 10 | `src/guardrails/input_guardrails.py` (shared `mask_phone`) | `tests/test_guardrails.py` | `pytest tests/test_guardrails.py::test_phone_masked_before_model_sees_input` (mock LLM records the prompt it received) | `transcripts/guardrail_demos.md` | Done |
| P2-16 | Prompt-injection guardrail demonstrated | 10 | `input_guardrails.py` | `tests/test_guardrails.py` | `pytest tests/test_guardrails.py::test_injection_blocked` | same | Done |
| P2-17 | Output groundedness refusal demonstrated | 10 | `src/guardrails/output_guardrails.py` | `tests/test_guardrails.py` | `pytest tests/test_guardrails.py::test_ungrounded_draft_refused` | same | Done |

### Part 3 — Evaluation, observability, FastAPI (20 marks)

| ID | Criterion | Task | Implementation | Test file | Verification command | Evidence artefact | Status |
|---|---|---|---|---|---|---|---|
| P3-01 | ≥2 working HTTP endpoints (`POST /ask`, `POST /add-document`) | 11 | `src/api/main.py`, `src/pipeline.py` | `tests/test_api.py` | `pytest tests/test_api.py -k "ask or add_document"` | `transcripts/api_demo.md` | Done |
| P3-02 | WebSocket `/chat` supports multi-turn chat | 11 | `src/api/main.py` | `tests/test_api.py` | `pytest tests/test_api.py::test_websocket_multi_turn` | same | Done |
| P3-03 | `WebSocketDisconnect` caught; server keeps serving others | 11 | `main.py` | `tests/test_api.py` | `pytest tests/test_api.py::test_websocket_disconnect_midchat_does_not_break_other_clients` | `transcripts/api_demo.md` | Done |
| P3-04 | Pydantic request/response models on HTTP payloads | 11 | `src/schemas/api.py` | `tests/test_api.py` | `pytest tests/test_api.py::test_pydantic_validation_rejects_malformed` (422 on empty/malformed) | — | Done |
| P3-05 | Every request → exactly one JSONL entry with trace ID, timestamp, latency; "ELK-style" = flat JSON keys, ISO-8601 `@timestamp`, `trace_id`, `duration_ms` (see X-29) | 12 | `src/observability/logging_config.py` | `tests/test_logging.py` | `pytest tests/test_logging.py::test_one_entry_per_request tests/test_logging.py::test_entry_has_trace_timestamp_latency` (includes rejected/errored requests and each WS turn) | `logs/sample_requests.jsonl` (generated, committed sample) | Done |
| P3-06 | Raw phone never reaches log file; masked form present; masking occurs pre-write | 12 | `logging_config.py`, shared `mask_phone` | `tests/test_logging.py` | `pytest tests/test_logging.py::test_raw_phone_absent_masked_present` (greps whole log file bytes) | same | Done |
| P3-07 | Exactly 15 queries covering all 12 KB topics + ≥2 OOS/edge | 13 | `evaluation/evaluation_dataset.py` | `tests/test_evaluation.py` | `pytest tests/test_evaluation.py::test_fifteen_queries_cover_twelve_topics_and_edges` | `evaluation/reports/15_query_results.md` | Done |
| P3-08 | Accuracy, Grounding, Completeness, Safety reported for every query; produced by an LLM-as-judge *prompt* sent through the shared `MockLLM` interface (X-32), not by an unrelated scoring function | 13 | `evaluation/judge.py`, `evaluation/run_evaluation.py` | `tests/test_evaluation.py` | `S: python -m evaluation.run_evaluation`; `pytest tests/test_evaluation.py::test_all_rows_have_four_scores` | same | Done |
| P3-09 | Four averages reported across all 15 queries | 13 | `run_evaluation.py` | `tests/test_evaluation.py` | `pytest tests/test_evaluation.py::test_four_averages_correct_and_reproducible` (run twice → identical) | same | Done |
| P3-10 | Evaluation + judge run under `MOCK_LLM`, no network | 13 | `judge.py` | `tests/test_evaluation.py`, `tests/test_offline.py` | `pytest tests/test_evaluation.py::test_runs_with_sockets_blocked` | — | Done |
| P3-11 | WebSocket handles concurrent connections with async patterns (`async def`, blocking work in a thread pool); multi-turn memory isolated per connection/session ID; no-session-ID connections get unique IDs | 11 | `src/api/main.py`, `src/memory/session_memory.py` | `tests/test_api.py` | `pytest tests/test_api.py::test_ws_concurrent_sessions_isolated` (≥2 clients, interleaved turns, each answer uses only its own session's context; same-ID connections deliberately share) | `transcripts/api_demo.md` | Done |

### Part 4 — Resilience and governance (20 marks)

| ID | Criterion | Task | Implementation | Test file | Verification command | Evidence artefact | Status |
|---|---|---|---|---|---|---|---|
| P4-01 | AutoGen team has exactly two agents, a policy-compliance reviewer and a final editor, named `Policy_Compliance_Reviewer` and `Final_Editor` (AutoGen rejects hyphenated names; the brief gives the names only as examples) | 14 | `src/review/autogen_review.py` | `tests/test_review.py` | `pytest tests/test_review.py::test_exactly_two_named_agents` | `transcripts/autogen_review_demos.md` | Done |
| P4-02 | `RoundRobinGroupChat` with `max_turns=2` (or `MaxMessageTermination(3)`) — team genuinely executes | 14 | `autogen_review.py` | `tests/test_review.py` | `pytest tests/test_review.py::test_round_robin_max_turns_two` (asserts exactly 2 agent messages after the task message) | same | Done |
| P4-03 | Final-Editor has `output_content_type=<VerdictModel>` | 14 | `autogen_review.py`, `src/schemas/verdict.py` | `tests/test_review.py` | `pytest tests/test_review.py::test_final_editor_output_content_type` | same | Done |
| P4-04 | Team registers `custom_message_types=[StructuredMessage[VerdictModel]]` | 14 | `autogen_review.py` | `tests/test_review.py` | `pytest tests/test_review.py::test_custom_message_types_registered` | same | Done |
| P4-05 | Review receives CrewAI draft **and** original retrieved context | 14 | `autogen_review.py`, `pipeline.py` | `tests/test_review.py` | `pytest tests/test_review.py::test_task_message_contains_draft_and_context` | same | Done |
| P4-06 | Verdict has `approved: bool`, `final_answer: str`, `reason: str` | 14 | `src/schemas/verdict.py` | `tests/test_review.py` | `pytest tests/test_review.py::test_verdict_schema_fields` | same | Done |
| P4-07 | Demo 1: draft approved unchanged | 14 | `autogen_review.py`, mock AutoGen client | `tests/test_review.py` | `pytest tests/test_review.py::test_supported_draft_approved_unchanged` | same | Done |
| P4-08 | Demo 2: injected ungrounded claim detected and removed (revision) | 14 | same | `tests/test_review.py` | `pytest tests/test_review.py::test_ungrounded_claim_revised_out` | same | Done |
| P4-09 | Only Lookup Agent has `check_job_application_status` | 15 | `src/governance/autonomy.py`, `agents.py` | `tests/test_governance_cache.py` | `pytest tests/test_governance_cache.py::test_only_lookup_agent_has_lookup_tool` | `transcripts/governance_autonomy_demo.md` | Done |
| P4-10 | Restriction demonstrated + one-paragraph explanation of enforcement | 15 | `autonomy.py`, `src/governance/risk_assessment.md` or `README.md` | `tests/test_governance_cache.py` | `pytest tests/test_governance_cache.py::test_other_agents_cannot_invoke_lookup` | same | Done |
| P4-11 | One explicit Low/Medium/High classification with a one-paragraph justification, using the brief's scheme (Low: summarization/transcription; Medium: code generation/customer-support tickets; High: medical, **hiring decisions**, financial). The system reads a candidate's expected salary (financial data) inside a hiring process, so it is classified High; it makes no hiring decision | 15 | `src/governance/risk_assessment.md` | `tests/test_governance_cache.py` | `pytest tests/test_governance_cache.py::test_risk_doc_has_single_classification` (exactly one of Low/Medium/High as the stated classification) + manual review | — | Done |
| P4-12 | Per-request token/cost budget cap enforced before crew execution | 15 | `src/governance/budget.py`, `pipeline.py` | `tests/test_governance_cache.py` | `pytest tests/test_governance_cache.py::test_budget_checked_before_crew` (crew never constructed on reject) | `transcripts/governance_budget_demo.md` | Done |
| P4-13 | Oversized request rejected; rejection demonstrated | 15 | `budget.py` | `tests/test_governance_cache.py` | `pytest tests/test_governance_cache.py::test_oversized_request_rejected_structured_error` | same | Done |
| P4-14 | Grounded-generation cache keyed by normalized query | 16 | `src/cache/grounded_cache.py` | `tests/test_governance_cache.py` | `pytest tests/test_governance_cache.py::test_normalized_key` (case/whitespace/punctuation variants collide; rule detail in P4-16) | `transcripts/cache_demo.md` | Done |
| P4-15 | Repeat query = cache hit, avoids redundant grounded-generation call, with before/after counter | 16 | `grounded_cache.py`, `grounded_generation.py` | `tests/test_governance_cache.py` | `pytest tests/test_governance_cache.py::test_repeat_query_hit_no_extra_generation_call` (counter 1→1, hit count 0→1) | same | Done |
| P4-16 | Cache-key normalization: strip leading/trailing whitespace and punctuation, lower-case, collapse internal whitespace, keep internal punctuation (record IDs); lexical only, no semantic-match claim; empty-after-normalization rejected before cache | 16 | `src/cache/grounded_cache.py` (`normalize_query`) | `tests/test_governance_cache.py` | `pytest tests/test_governance_cache.py::test_normalize_query_rules` (`"  What is the NOTICE period? "`, `"what is the notice period"`, `"what is the notice period?!"` collide; `APP-0007` keeps its hyphen; a paraphrase misses) | `transcripts/cache_demo.md` | Done |

### Submission

| ID | Criterion | Implementation | Test / check | Verification command | Status |
|---|---|---|---|---|---|
| S-01 | One public GitHub repo with all code, KB, dataset, tests, reports, transcripts | repo root | manual | `git ls-files` review; `gh repo view --json visibility` after push (**requires user action to publish**) | Requires the repository owner (publish as public) |
| S-02 | README begins by identifying Naukri.com (Recruitment & HR) track | `README.md` | `tests/test_readme_consistency.py` | `pytest tests/test_readme_consistency.py::test_track_stated_first` | Done |
| S-03 | README documents exact dataset-generation choices | `README.md` | same | `pytest tests/test_readme_consistency.py::test_dataset_choices_match_code` | Done |
| S-04 | README confirms zero API keys / zero network for graded `MOCK_LLM` runs | `README.md` | `tests/test_offline.py`, readme test | `pytest tests/test_offline.py tests/test_readme_consistency.py::test_readme_confirms_offline` | Done |
| S-05 | README confirms CrewAI telemetry disabled | `README.md` | readme test | `pytest tests/test_readme_consistency.py::test_readme_confirms_telemetry` | Done |
| S-06 | README contains measured similarity values + chosen threshold | `README.md` | readme test | `pytest tests/test_readme_consistency.py::test_readme_similarity_matches_report` | Done |
| S-07 | No prohibited artefacts (images, PDFs, decks, video, audio) | whole repo | `tests/test_repo_hygiene.py` | `pytest tests/test_repo_hygiene.py::test_no_prohibited_file_types` (also flags `.DS_Store`) | Done |

---

## 2b. Clause-by-clause coverage of the capstone brief

Every sentence-level obligation in the capstone brief, in order, with the criterion/requirement that covers it. "—" in the third column would mean uncovered; there are none.

| ID | Brief clause | Covered by |
|---|---|---|
| B-01 | One production-minded agent combining dataset, embeddings, retrieval, grounded generation, memory, tool-calling, multi-agent orchestration, structured outputs, guardrails, evaluation, governance, cost control, API | all Parts; X-27 (one end-to-end test touching every block) |
| B-02 | Answers policy questions from a self-written KB | P1-06, P1-07, P1-13 |
| B-03 | Looks up real records from a self-designed, self-validated dataset | P1-01…P1-05, P2-01 |
| B-04 | Remembers a conversation | P2-12, P2-13 |
| B-05 | Guarded against misuse | P2-15…P2-17 |
| B-06 | Answers reviewed by a second, **independent** agent team before reaching the user | P4-01…P4-08, X-05 |
| B-07 | Explicit governance policy | P4-09…P4-13 |
| B-08 | Orchestrated with CrewAI | P2-04, P2-05 |
| B-09 | Deployed behind FastAPI | P3-01…P3-04 |
| B-10 | Evaluated end to end | P3-07…P3-10 (evaluation runs the full pipeline) |
| B-11 | Standard of "could pass a real production governance review, not a happy-path demo" | P4-09…P4-13, X-03, X-07, X-11, risk assessment doc |
| B-12 | One deliverable, single public repo | S-01 |
| B-13 | One link; no per-Part submission | S-01 |
| B-14 | README top: states Naukri.com (Recruitment & HR) track | S-02 |
| B-15 | README top: exact Task 1 choices — seed, category/status weights, amount range | S-03, P1-05 |
| B-16 | No images/screenshots/uploaded diagrams/PDFs/decks/video/audio anywhere | S-07 |
| B-17 | No paid account or credit card | X-22, X-11 (all deps free/local) |
| B-18 | SentenceTransformers + ChromaDB free and local | P1-10, P1-11 |
| B-19 | LLM calls of **four** consumers — grounded generation, evaluation judge, CrewAI crew, AutoGen review team — default to deterministic `MOCK_LLM`, zero keys, zero network | X-23, X-11 |
| B-20 | Graded transcripts use `MOCK_LLM` | X-21, X-31 |
| B-21 | Real LLM optional behind env flag; all criteria satisfied under mock alone | X-22 |
| B-22 | Extend `crewai.llms.base_llm.BaseLLM`, no external interception | P2-08 |
| B-23 | Pitfall 1: don't match template `Observation:` | P2-09 |
| B-24 | Pitfall 2: don't dispatch by tool-name substring (`rag_lookup`); use declared argument schema | P2-10 |
| B-25 | Set `CREWAI_DISABLE_TELEMETRY=true` (or `OTEL_SDK_DISABLED=true`) before the crew runs; confirm in README | P2-11, S-05 |
| B-26 | Official docs may be consulted | n/a (permission, not obligation) |
| B-27 | Originality: dataset, KB, code, analysis are own work for this brief | X-24 |
| B-28 | Submit by end of the 14-day window | process item; tracked in status doc |
| B-29 | Category vocabulary (5 values; each ≥3 records per Task 1) | P1-02 |
| B-30 | Status vocabulary (5 values; each ≥1 record) | P1-03 |
| B-31 | 12 required KB topics, ≥12 docs, 2–5 sentences | P1-06, P1-07 |
| B-32 | Phone number in contact details is fixed-format and demonstrated masked; name, expected salary, background-check results out of scope (not claimed); fabricated examples only for all three | P2-15, X-13, X-25 |
| B-33 | Task 1: seeded deterministic generator, ≥40 records, six fields, salary range with one-sentence reasoning, counts per category/status, flagged % in 10–30, no hand-editing | P1-01…P1-05, X-20 |
| B-34 | Task 2: ≥12 docs, 2–5 sentences, own words | P1-06, P1-07, X-24 |
| B-35 | Task 3: both chunkings, local embeddings, separate Chroma collections via `upsert()` | P1-08…P1-11 |
| B-36 | Task 4: top-k from either collection; context-only generation; similarity as only signal under mock; calibrate with ≥3 in-scope and ≥2 out-of-scope; no untested 0.5/0.6/0.7; README values + threshold; ≥5 in-scope + 1 OOS fallback | P1-12…P1-14 |
| B-37 | Task 5: same ≥5 queries; doc-level P/R with dedup; per-query arithmetic for both; 2–3 sentence numbers-cited recommendation | P1-15…P1-17 |
| B-38 | Task 6: tool signature, three returned fields, designed score in [0,1] combining flag + normalized recency, exact formula, threshold justified from own distribution | P2-01…P2-03 |
| B-39 | Task 7: ≥3 agents (Retrieval w/ RAG tool, Lookup w/ lookup tool, Composer), `.kickoff()`, both tools invoked on different queries | P2-04…P2-07 |
| B-40 | Task 8: LangChain session memory; multi-turn transcript; separate fresh-conversation transcript; in-process only; deprecation warning not silenced | P2-12, P2-13, X-26 |
| B-41 | Task 9: Pydantic model as `response_format`; every response validated in code | P2-14, X-28 |
| B-42 | Task 10: input PII masking, prompt-injection detection, output groundedness refusal; each demonstrated firing on one deliberate case | P2-15…P2-17 |
| B-43 | Task 11: ≥2 HTTP endpoints with Pydantic models + WebSocket multi-turn, `WebSocketDisconnect` caught, server keeps serving others | P3-01…P3-04, P3-11, X-18 |
| B-44 | Task 12: one JSONL entry per request with trace ID and timing, ELK-style; logged text has no raw masked-PII value; same masking for model and log | P3-05, P3-06, X-29 |
| B-45 | Task 13: LLM-as-judge under `MOCK_LLM`; 15 queries; ≥1 per KB topic; ≥2 OOS/edge; four scores per query + four averages | P3-07…P3-10, X-32 |
| B-46 | Task 14: `RoundRobinGroupChat`, `max_turns=2` (or `MaxMessageTermination(3)`), `output_content_type`, `custom_message_types`, verdict fields, draft + original context in, approve-unchanged and revise demos | P4-01…P4-08, X-30 |
| B-47 | Task 15: only Lookup Agent may call the lookup tool, demonstrated, one-paragraph guard explanation; risk class + one-paragraph justification; runtime per-request budget cap, oversized request rejected | P4-09…P4-13 |
| B-48 | Task 16: in-memory cache keyed by normalized query for grounded generation; repeat query → hit, avoids redundant LLM/tool call, before/after evidence | P4-14…P4-16, X-07 |
| B-49 | Acceptance criteria Parts 1–4 and Submission bullets | §2 (each bullet is an AC row; the requirements spec's §10 bullets are a superset) |
| B-50 | Repo contains `dataset.py`, KB docs, RAG/crew/review/governance/deployment code, **transcripts demonstrating every task**, README confirming `MOCK_LLM` with zero API keys | S-04, X-21, X-31 |
| B-51 | Exactly one submission link | S-01 |

---

## 3. Requirements outside the §10 checklist (must not be dropped)

These come from the requirements body or architecture and are verifiable, but have no checkbox. They are tracked so they are not silently skipped.

| ID | Requirement (source) | Implementation | Verification | Status |
|---|---|---|---|---|
| X-01 | README top: escalation formula + threshold (§1.2 item 6) | `README.md` | `pytest tests/test_readme_consistency.py::test_readme_escalation_formula` | Done |
| X-02 | README setup/run/test instructions reproducible from a clean clone (§1.2 item 7, §8.1) | `README.md`, `scripts/run_acceptance.sh` | fresh-venv run of `scripts/run_acceptance.sh` | Done |
| X-03 | Missing application IDs / empty / malformed requests handled (§8.3) | `application_lookup.py`, `schemas/api.py` | `tests/test_api.py::test_empty_and_malformed`, `tests/test_escalation.py::test_unknown_id` | Done |
| X-04 | Unsupported policy questions fall back (§8.3) | `grounded_generation.py` | covered by P1-14 + `tests/test_api.py::test_ask_out_of_scope` | Done |
| X-05 | AutoGen review runs after Composer and before delivery (§8.3) | `pipeline.py` | `pytest tests/test_review.py::test_pipeline_order_review_after_composer_before_return` (call-order recorder) | Done |
| X-06 | Pipeline order: validate → mask → inject-detect → budget → memory → crew → validate → groundedness → review → final validate → memory update → log (arch §9.1) | `pipeline.py` | `tests/test_governance_cache.py::test_stage_order_recorded` | Done |
| X-07 | Cache never bypasses guardrails, budget, memory, schema validation, review (§7.3, arch §9.2) | `pipeline.py`, `grounded_cache.py` | `tests/test_governance_cache.py::test_cache_hit_still_runs_all_controls` | Done |
| X-08 | Tool-invocation evidence in transcripts/logs (§5.2, §8.4) | `rag_tool.py`, `application_lookup.py` (invocation recorder) | covered by P2-06/07 | Done |
| X-09 | Cache hit/miss evidence in logs (§8.4) | `logging_config.py` | `tests/test_logging.py::test_log_records_cache_hit_miss` | Done |
| X-10 | Budget estimate + cap in logs/transcripts; stated as simulated, no real charge (§7.2, plan §11.3) | `budget.py`, `logging_config.py` | `tests/test_governance_cache.py::test_budget_fields_in_log` | Done |
| X-11 | **No network calls in graded mode** — enforced, not asserted (§1.1, §5.3) | `tests/conftest.py` socket-block fixture, `src/config/settings.py` | `pytest` entire suite under socket block; `tests/test_offline.py::test_outbound_socket_raises` | Done |
| X-12 | Chroma/HF/LangSmith telemetry also off (arch §2.3 "no external telemetry") | `settings.py` | `tests/test_offline.py::test_all_telemetry_flags_off` | Done |
| X-13 | Do not claim name/salary/background-check masking; README states the limit (§1.1, §2.2) | `README.md` | `tests/test_readme_consistency.py::test_readme_states_masking_scope` | Done |
| X-14 | Injection detector scope & limitations stated (§5.6) | `input_guardrails.py` docstring, `README.md` | readme test | Done |
| X-15 | Judge rubric + score scale explicit before running (§6.3, plan §9) | `evaluation/judge.py`, report header | `tests/test_evaluation.py::test_report_states_rubric_and_scale` | Done |
| X-16 | Judge is not claimed to be a semantic LLM judge (arch §3.3, plan §9) | report header, README | readme/report test | Done |
| X-17 | `/add-document` is additive; original 12 docs unchanged; stable IDs (arch §4.3) | `main.py`, `vector_store.py` | `tests/test_api.py::test_add_document_additive` | Done |
| X-18 | Per-connection resources released on disconnect; shared in-process state (session store, cache, log writer) safe under concurrent access (arch §4.4 steps 7, 10) | `session_memory.py`, `main.py`, `grounded_cache.py` | `tests/test_api.py::test_ws_disconnect_releases_resources`, `tests/test_governance_cache.py::test_cache_thread_safe` | Done |
| X-19 | `.env.example` has no secrets; no keys/secrets logged (§8.2) | `.env.example`, `logging_config.py` | `tests/test_repo_hygiene.py::test_no_secrets_in_env_example_or_logs` | Done |
| X-20 | Unique, stable record IDs; `days_since_created` is int in [0,30]; flag is bool; salary in documented range (plan §4.1) | `dataset.py` | `tests/test_dataset.py::test_field_types_and_ranges` | Done |
| X-21 | Every required transcript exists (§8.4, §9, arch §10): distribution, calibration, RAG demos, P/R, tool invocation, memory, guardrails, WS disconnect, 15-query eval, AutoGen approve/revise, autonomy, budget, cache | `scripts/generate_transcripts.py` | `pytest tests/test_repo_hygiene.py::test_all_required_transcripts_exist_and_nonempty` | Done |
| X-22 | Real-LLM mode optional only; never required (§1.1; brief: "optionally wired in behind an environment-variable flag") | `settings.py` | default `MOCK_LLM=true`; `MOCK_LLM=false` raises a clear "real LLM path not implemented" error | Done |
| X-23 | One `MOCK_LLM` switch governs all **four** LLM consumers named in the brief: grounded generation, evaluation judge, CrewAI crew, AutoGen review team | `src/llm/mock_llm.py` (shared deterministic core), `mock_crewai_llm.py`, `mock_autogen_client.py`, judge, grounded generation | `tests/test_offline.py::test_all_four_consumers_run_mock_with_sockets_blocked` | Done |
| X-24 | Originality: dataset, KB, code, analysis authored for this brief; nothing copied from the brief/specs or external sources | `knowledge_base/`, `README.md` | `tests/test_kb.py::test_kb_text_not_copied_from_specs` (no ≥8-word shared run with `specs/*.md`) + README originality statement + manual review | Done |
| X-25 | Fabricated-only data for candidate name, salary, background-check and phone examples | fixtures, transcripts | `tests/test_repo_hygiene.py::test_phone_fixtures_are_fabricated` (all phone examples from a documented fake-number list) | Done |
| X-26 | `LangChainDeprecationWarning` from `RunnableWithMessageHistory` is expected and **not silenced** | `pytest.ini`, `session_memory.py` | `tests/test_memory.py::test_deprecation_warning_not_suppressed` (warning observed under `pytest.warns`) | Done |
| X-27 | One end-to-end test exercises every building block in a single request (mask → inject-check → budget → memory → crew w/ both tools → schema → groundedness → AutoGen → log) | `src/pipeline.py` | `tests/test_pipeline_e2e.py::test_full_request_touches_every_block` | Done |
| X-28 | `response_format`: Composer task declares the Pydantic model (CrewAI-native structured output) *and* the pipeline validates in code | `crew/tasks.py`, `schemas/crew_response.py` | covered by P2-14; spike S2 records which CrewAI parameter works with a custom `BaseLLM` | Done |
| X-29 | ELK-style logging: one flat JSON object per line; ISO-8601 `@timestamp`; `trace_id`; `duration_ms`; `level`; `event`/`endpoint`; `status`; no nested free-form dumps | `observability/logging_config.py` | `tests/test_logging.py::test_elk_style_fields_flat_and_parseable` | Done |
| X-30 | AutoGen: building the team *without* `custom_message_types` fails with `ValueError: Message type … is not registered`; test documents why registration is required | `autogen_review.py` | `tests/test_review.py::test_missing_registration_raises_valueerror` | Done |
| X-31 | Transcripts demonstrate **every** task 1–16 (see §3b) | `scripts/generate_transcripts.py` | `tests/test_repo_hygiene.py::test_transcript_for_every_task` | Done |
| X-32 | LLM-as-judge: a judge *prompt* (rubric, 1–5 scale, four properties, JSON output) is sent through the shared `MockLLM`; the mock returns deterministic JSON scores computed from per-query expected-evidence fields | `evaluation/judge.py`, `src/llm/mock_llm.py` | `tests/test_evaluation.py::test_judge_goes_through_mock_llm_with_prompt` | Done |

---

## 3b. Transcript / report coverage per task

The brief requires "transcripts demonstrating every task above". Each is generated by `scripts/generate_transcripts.py` (or the named script).

| Task | Artefact(s) |
|---|---|
| 1 Dataset | `transcripts/dataset_distribution.txt` (counts per category/status, flagged %) |
| 2 Knowledge base | `transcripts/kb_validation.txt` (12 docs, sentence counts, topic→doc-ID map) |
| 3 Chunking + indexing | `transcripts/rag_indexing.md` (chunk counts per strategy, collection names, `upsert` counts, one sample query on **each** collection) |
| 4 Grounded generation | `transcripts/rag_demos.md` (≥5 in-scope + 1 fallback), `evaluation/reports/similarity_calibration.md` |
| 5 Chunking comparison | `evaluation/reports/chunking_comparison.md` |
| 6 Lookup + escalation | `evaluation/reports/escalation_threshold.md`, `transcripts/crew_tool_invocations.md` (known + unknown ID) |
| 7 Crew | `transcripts/crew_tool_invocations.md` (RAG query run, lookup query run) |
| 8 Memory | `transcripts/memory_sessions.md` (Session A multi-turn; Session B fresh) |
| 9 Structured output | `transcripts/structured_output.md` (valid responses + a rejected malformed one) |
| 10 Guardrails | `transcripts/guardrail_demos.md` (phone, injection, groundedness) |
| 11 FastAPI | `transcripts/api_demo.md` (`/ask`, `/add-document`, WS multi-turn, mid-chat disconnect, second client unaffected) |
| 12 Logging | `logs/sample_requests.jsonl`, `transcripts/logging_privacy_demo.md` |
| 13 Evaluation | `evaluation/reports/15_query_results.md` (15 rows × 4 scores + 4 averages) |
| 14 AutoGen | `transcripts/autogen_review_demos.md` (approve + revise) |
| 15 Governance | `transcripts/governance_autonomy_demo.md`, `src/governance/risk_assessment.md`, `transcripts/governance_budget_demo.md` |
| 16 Cache | `transcripts/cache_demo.md` |

---

## 4. Mapping summary

| Part | Criteria | Done | Requires the repository owner |
|---|---|---|---|
| 1 | 17 | 17 | 0 |
| 2 | 17 | 17 | 0 |
| 3 | 11 | 11 | 0 |
| 4 | 16 | 16 | 0 |
| Submission | 7 | 6 | 1 (S-01) |
| Extra (X) | 32 | 32 | 0 |
| **Total** | **100** | **99** | **1** |

Plus 51 brief-clause rows (§2b) that all resolve to the rows above.

The release gate (Requirements §11) is the pass/fail matrix generated by `scripts/acceptance_report.py` from a JUnit report and the criterion IDs tagged on tests with `@pytest.mark.ac("P1-01")`; it never marks a criterion passed without an executed test. Run everything with `bash scripts/run_acceptance.sh`.
