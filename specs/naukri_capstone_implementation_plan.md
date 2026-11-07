# Naukri.com Domain Support Agent — Implementation Plan

**Project:** Naukri.com Domain Support Agent  
**Baseline requirements:** `naukri_capstone_requirements.md`  
**Architecture:** `naukri_capstone_architecture.md`  
**Execution target:** Local, offline, deterministic `MOCK_LLM`  
**Submission:** One public GitHub repository  
**Timebox:** 14 days (estimated capstone duration)

---

## 1. Delivery strategy

Implement the application incrementally, one phase at a time, never the entire application in one step.

The implementation must follow dependency order:
1. Validate requirements and repository plan.
2. Build deterministic data and knowledge base.
3. Build and measure RAG.
4. Prototype CrewAI's custom mock LLM and tool calling.
5. Build crew, memory, structured output and guardrails.
6. Add FastAPI, WebSocket and safe logging.
7. Build the 15-query evaluation.
8. Add AutoGen review.
9. Add governance and cache.
10. Run the complete offline acceptance suite and prepare the README/repository.

Each phase has:
- inputs,
- implementation work,
- outputs,
- tests/exit gates,
- evidence to retain.

Do not advance to the next phase while a required exit gate is failing.

## 2. Proposed 14-day schedule

| Day | Workstream | Completion gate |
|---|---|---|
| 1 | Repository skeleton, dependency compatibility checks, config and traceability matrix | Clean environment imports required frameworks; requirements mapped to modules/tests |
| 2 | Dataset generator and validation | At least 40 records; exact categories/statuses; priority percentage in range |
| 3 | Author all 12 policy documents and KB validation | 12 documents; each 2–5 sentences; exact topic coverage |
| 4 | Chunkers, local embeddings and ChromaDB collections | Both collections indexed and queryable |
| 5 | Similarity calibration, grounded generation, precision/recall comparison | 5+1 RAG demos; empirical threshold; per-query metrics and recommendation |
| 6 | Lookup tool, escalation formula and CrewAI mock LLM compatibility spike | Lookup tested; BaseLLM works; tool dispatch pitfalls tested |
| 7 | CrewAI agents, tasks, `.kickoff()` and Pydantic output | 3-agent crew works; both tools invoked on separate examples |
| 8 | Session memory and input/output guardrails | Multi-turn/fresh-session transcripts; PII, injection and groundedness tests pass |
| 9 | FastAPI HTTP endpoints and Pydantic API contracts | `/ask` and `/add-document` pass integration tests |
| 10 | WebSocket, disconnect handling and JSONL logging | Multi-turn WebSocket works; disconnect safe; no raw phone in logs |
| 11 | 15-query deterministic evaluation suite | All queries scored on four metrics; averages generated |
| 12 | AutoGen two-agent structured review | Offline team passes approval and revision demonstrations |
| 13 | Least autonomy, risk classification, budget and cache | Tool restriction, oversized rejection and cache-hit evidence pass |
| 14 | Full offline regression, README, transcripts and GitHub readiness | All acceptance criteria pass; one repository ready to submit |

If package compatibility or local model setup takes longer, reduce optional features rather than weakening mandatory acceptance criteria.

---

## 3. Phase 0 — Repository setup and dependency compatibility

### Inputs
- `naukri_capstone_requirements.md`
- `naukri_capstone_architecture.md`
- This implementation plan.

### Work
1. Create the repository and virtual environment.
2. Establish the proposed module layout.
3. Install candidate versions of required libraries.
4. Verify imports and minimal examples for:
   - CrewAI `BaseLLM` subclass;
   - AutoGen `RoundRobinGroupChat`;
   - AutoGen `StructuredMessage` registration and `output_content_type`;
   - LangChain `InMemoryChatMessageHistory` and `RunnableWithMessageHistory`;
   - FastAPI WebSocket;
   - SentenceTransformers local model;
   - ChromaDB collection creation and `upsert()`.
5. Run a minimal offline startup check after the embedding model is locally available.
6. Pin versions only after the compatibility checks pass.
7. Configure environment variables before importing/running the crew:
   - `CREWAI_DISABLE_TELEMETRY=true` or `OTEL_SDK_DISABLED=true`;
   - `MOCK_LLM=true`;
   - any local model/cache path configuration required by the implementation.

### Outputs
- Repository skeleton.
- Dependency manifest with tested versions.
- `.env.example` with no secrets.
- `src/config/settings.py` or equivalent.
- `docs/requirements_traceability.md` mapping every task/acceptance criterion to code and tests.

### Exit gate
A fresh environment can import the required libraries and run the compatibility smoke tests. No API key is required.

---

## 4. Phase 1 — Dataset and knowledge base

### 4.1 Dataset generator

Implement `dataset.py`.

Suggested design:
- Use a fixed documented seed.
- Generate at least 40 records.
- Guarantee required category/status coverage through deterministic generation design, not post-generation manual edits.
- Generate priority flags using a documented probability/weight and validate the resulting percentage.
- Use a realistic expected-salary range and document the rationale.
- Generate integer `days_since_created` from 0 to 30 inclusive.
- Use stable unique record IDs.

Validation assertions:
- `len(records) >= 40`
- all required category values are present;
- each required category count is at least 3;
- all required status values are present;
- each required status count is at least 1;
- every `days_since_created` value is an integer in `[0, 30]`;
- every priority flag is boolean;
- priority percentage is in `[10, 30]`;
- record IDs are unique;
- salary values fall within the documented selected range.

Print category counts, status counts and priority percentage.

### 4.2 Knowledge-base documents

Create one file per required policy topic:
1. job-application eligibility;
2. interview scheduling;
3. offer negotiation;
4. background verification;
5. notice period;
6. referral bonus;
7. internal transfer;
8. probation period;
9. remote work;
10. diversity hiring;
11. exit interview;
12. applicant-data retention.

Each must contain 2–5 complete sentences, and must cover exactly the required topic. Create a loader that checks the number of documents and sentence-count range.

### Outputs
- `dataset.py`
- Dataset validation tests.
- 12+ KB Markdown documents.
- KB loader and KB validation report.
- README dataset choices.

### Exit gate
Dataset and KB checks pass from a clean run.

---

## 5. Phase 2 — RAG implementation and evaluation

### 5.1 Chunking

Implement two independent chunkers:
- fixed-size chunks with overlap;
- sentence-based chunks.

Every chunk must preserve:
- stable chunk ID;
- parent document ID;
- source/title;
- chunking strategy.

Test edge cases: short document, long document, one-sentence document, punctuation, empty text and duplicate indexing.

### 5.2 Embedding and ChromaDB

1. Select a free local SentenceTransformers model.
2. Make the model available locally before offline evaluation.
3. Embed all chunks from both strategies.
4. Create separate persistent ChromaDB collections.
5. Upsert chunks with stable IDs and metadata.
6. Confirm that repeated indexing does not create unwanted duplicates.
7. Confirm each collection returns sensible top-k results.

### 5.3 Similarity calibration

Prepare at least:
- 3 in-scope policy queries;
- 2 deliberately out-of-scope queries.

Record top-1 cosine similarity for every calibration query. Choose a threshold between the observed in-scope and out-of-scope clusters. If the clusters overlap, do not hide the problem: refine the query set/model/chunking and document the measured outcome before selecting a defensible threshold.

The chosen threshold and raw values must appear in README and a machine-readable or Markdown evaluation report.

### 5.4 Grounded generation

Implement:
- retrieval of top-k chunks;
- context assembly;
- deterministic mock answer generation using only retrieved text;
- explicit fallback when similarity is below the calibrated threshold;
- source document IDs in the result.

Demonstrate five or more in-scope questions and one out-of-scope fallback.

### 5.5 Document-level precision and recall

For at least five common in-scope queries:
1. Declare relevant parent-document IDs before scoring.
2. Retrieve top-k chunks from both collections.
3. Map chunks to parent document IDs.
4. Deduplicate parent IDs.
5. Calculate precision and recall for each query/collection.
6. Show arithmetic, not only the final score.
7. Compare aggregates and recommend a strategy in 2–3 sentences with actual numbers.

### Outputs
- `src/rag/chunking.py`
- `src/rag/embeddings.py`
- `src/rag/vector_store.py`
- `src/rag/retrieval.py`
- `src/rag/grounded_generation.py`
- RAG calibration and comparison reports.
- Five-plus-one demonstration transcript.

### Exit gate
Both collections work; empirical fallback works; all required RAG metrics and evidence exist.

---

## 6. Phase 3 — Lookup tool and CrewAI mock LLM spike

### 6.1 Application lookup

Implement:

`check_job_application_status(record_id: str) -> dict`

Return:
- `status`;
- `expected_salary_inr`;
- `escalation_score`.

Choose and document a formula combining the priority flag and normalized recency. A proposed formula is:

`score = 0.60 * int(flagged_priority_review) + 0.40 * (1 - days_since_created / 30)`

Validate that the result is always between 0 and 1. Calculate the actual dataset's `days_since_created` percentiles (P20, P50, P80) and justify the recommended escalation threshold from them. Do not hardcode a claimed percentile without calculating it.

Test:
- known record;
- unknown record;
- youngest/oldest recency values;
- priority flag true/false;
- score boundaries.

### 6.2 CrewAI custom `BaseLLM` compatibility spike

Do this before creating all agents.

1. Subclass `crewai.llms.base_llm.BaseLLM`.
2. Implement the methods required by the installed CrewAI version.
3. Return deterministic outputs for mock prompts.
4. Support the tool-call format expected by CrewAI.
5. Dispatch arguments using each tool's declared schema.
6. Parse generated model text rather than searching the whole system prompt for `Observation:`.
7. Explicitly test a tool named `rag_lookup` so name-substring dispatch cannot misroute it.
8. Test first-turn behaviour where no tool observation exists.
9. Confirm `.kickoff()` works without network access.
10. Confirm telemetry is disabled before execution.

### Outputs
- `src/tools/application_lookup.py`
- `src/llm/mock_crewai_llm.py`
- Compatibility tests and transcript.

### Exit gate
Both the RAG-style tool and application lookup tool can be called correctly by the mock CrewAI LLM without network access.

---

## 7. Phase 4 — CrewAI crew, structured output, memory and guardrails

### 7.1 Crew construction

Create:
- Retrieval Agent with RAG tool only.
- Lookup Agent with application lookup tool only.
- Response Composer with no application lookup tool.

Create tasks and a crew. Run with `.kickoff()`.

Demonstrate:
- policy query invokes RAG tool;
- application-status query invokes lookup tool;
- composer combines tool result(s) into a final draft.

### 7.2 Structured output

Define a Pydantic response model. Ensure the final crew output is parsed and validated in code for every request. Invalid output must fail safely or be corrected by a controlled deterministic validation path; do not silently return malformed structures.

### 7.3 Session memory

Use `InMemoryChatMessageHistory` and `RunnableWithMessageHistory` as required.

Demonstrate two transcripts:
- Session A: turn 1 establishes context; turn 2 correctly refers to it.
- Session B: a fresh session cannot see Session A's context.

Keep memory in process only.

### 7.4 Guardrails

Implement and test:
- fixed-format phone number masking;
- prompt-injection detection and rejection/neutralization;
- output groundedness refusal for unsupported content.

Ensure PII masking happens before the model sees the request. Reuse the same masking function for logging.

Document that names, expected salary and background-check results are not masked by the specified keyless masker and use fabricated examples only.

### Outputs
- `src/crew/agents.py`
- `src/crew/tasks.py`
- `src/crew/crew.py`
- `src/schemas/crew_response.py`
- `src/memory/`
- `src/guardrails/`
- Crew, memory and guardrail transcripts.

### Exit gate
Crew tool invocation, schema validation, memory reset and all guardrail demonstrations pass.

---

## 8. Phase 5 — FastAPI, WebSocket and logging

### 8.1 HTTP routes

Implement:
- `POST /ask`
- `POST /add-document`

Use Pydantic request and response models. Return controlled validation and error responses.

### 8.2 WebSocket

Implement `@app.websocket("/chat")`.
- Support multi-turn messages.
- Associate each connection/session with an appropriate session ID.
- Reuse input guardrails, budget checks, crew, review and output validation.
- Catch `WebSocketDisconnect`.
- Do not allow a disconnect to terminate the server or disrupt other clients.
- Use an `async def` handler and keep blocking pipeline work (CrewAI, AutoGen) off the event loop, for example with a thread pool.
- Keep multi-turn memory isolated per connection/session ID; a connection without a session ID gets a unique one.
- Test two or more concurrent clients with interleaved turns, each seeing only its own session's context.

### 8.3 Structured logging

Write exactly one JSONL request record per request. Include trace ID and timing.

Critical privacy test:
1. Send a fabricated fixed-format phone number.
2. Confirm model input contains the masked value.
3. Confirm the raw number does not appear anywhere in the log file.
4. Confirm the masked representation is present.
5. Confirm exactly one request log event was created.

Use middleware or a request-scoped `try/finally` design to ensure errors/rejections are logged too. Avoid duplicate logging between middleware and endpoint code.

### Outputs
- `src/api/main.py`
- API Pydantic schemas.
- `src/observability/logging_config.py`
- HTTP, WebSocket and log privacy tests.

### Exit gate
All three endpoints work; disconnect is safe; one PII-safe JSONL record is produced per request.

---

## 9. Phase 6 — Fifteen-query evaluation suite

Create exactly 15 evaluation queries:
- at least one query for each of the 12 required KB topics;
- at least 2 out-of-scope/edge-case queries;
- one remaining query may test another supported scenario.

Implement a deterministic LLM-as-judge interface under `MOCK_LLM`. It may use explicit rubric rules and evidence checks, but must not claim to be a live semantic LLM judge.

For every query, report:
- query ID and text;
- expected evidence/answer criteria;
- Accuracy;
- Grounding;
- Completeness;
- Safety;
- brief scoring rationale.

Calculate and report the average of each metric across all 15 queries. Define the score scale and rubric before running the evaluation.

### Outputs
- `evaluation/evaluation_dataset.py`
- `evaluation/judge.py`
- `evaluation/run_evaluation.py`
- `evaluation/reports/15_query_results.md` or `.json`
- Four metric averages.

### Exit gate
All 15 queries have all four scores and the four averages are reproducible.

---

## 10. Phase 7 — AutoGen review stage

### 10.1 Compatibility spike first

Before wiring the review into the API:
1. Verify installed AutoGen APIs.
2. Instantiate a two-agent `RoundRobinGroupChat`.
3. Configure `max_turns=2`.
4. Define the Pydantic verdict model with `approved`, `final_answer`, `reason`.
5. Set `output_content_type` on Final-Editor.
6. Register `StructuredMessage[VerdictModel]` through `custom_message_types`.
7. Run one offline mock review and validate the returned structured verdict.

Use a deterministic local/mock model integration compatible with the installed AutoGen version. The required AutoGen team must actually execute; do not substitute a hand-written review function that bypasses it.

### 10.2 Review behaviour

Input:
- CrewAI Composer draft;
- original retrieved context.

Output:
- structured verdict;
- unchanged answer when supported;
- revised answer when a deliberate unsupported claim is present.

Demonstrate at least:
- one unchanged approval;
- one revision that removes the intentionally injected ungrounded claim.

### Outputs
- `src/review/autogen_review.py`
- Verdict schema.
- Compatibility tests.
- Approval/revision transcript.

### Exit gate
AutoGen team executes offline and both review cases produce valid structured verdicts.

---

## 11. Phase 8 — Governance and caching

### 11.1 Least autonomy

Only Lookup Agent receives `check_job_application_status`.

Add a test that inspects the configured tools:
- Lookup Agent includes the lookup tool.
- Retrieval Agent does not include it.
- Response Composer does not include it.

Document the control and the reason for it.

### 11.2 Risk classification

Use the Low/Medium/High definitions given in the requirements document. Classify the implemented support agent based on actual capabilities. Explain whether it merely answers policy/status questions or makes consequential hiring decisions. Do not describe nonexistent salary ranking or automated hiring decisions.

### 11.3 Runtime budget

Implement a per-request token/cost budget cap. Under mock mode:
- use a deterministic token estimate;
- configure a maximum;
- reject over-limit requests before crew execution;
- include the estimated count, configured limit and rejection status in safe logs/transcripts;
- explain that no actual provider charge occurs in mock mode.

### 11.4 Grounded-generation cache

Implement in-memory caching keyed by normalized query text. Normalization: strip leading/trailing whitespace and punctuation, convert to lower case, and collapse internal whitespace runs to one space; preserve internal punctuation (for example record IDs). It is lexical only; make no semantic-match claim. Reject queries that are empty after normalization at input validation.

Demonstrate:
1. First request: cache miss; grounded-generation call counter increments.
2. Second identical/normalized request (for example differing only in case, surrounding whitespace or trailing punctuation): cache hit; counter does not increment.
3. Both outputs remain schema-valid and pass safety/review controls.

Do not cache the complete final answer in a way that bypasses the required per-request controls or review.

### Outputs
- `src/governance/autonomy.py`
- `src/governance/budget.py`
- `src/governance/risk_assessment.md`
- `src/cache/grounded_cache.py`
- governance/cache tests and transcripts.

### Exit gate
Tool isolation, risk justification, budget rejection and cache hit evidence all pass.

---

## 12. Phase 9 — Full integration and release validation

Run all tests in a clean environment with:
- `MOCK_LLM=true`;
- CrewAI telemetry disabled;
- no API keys;
- no network access;
- local embedding model already available.

Run:
1. Dataset generation and validation.
2. KB validation.
3. Both RAG indexing strategies.
4. Similarity calibration.
5. RAG in-scope and out-of-scope demos.
6. Precision/recall comparison.
7. Application lookup tests.
8. Crew `.kickoff()` tool invocation tests.
9. Memory tests.
10. Guardrail tests.
11. HTTP endpoint tests.
12. WebSocket multi-turn and disconnect tests.
13. JSONL PII privacy test.
14. Fifteen-query evaluation.
15. AutoGen approval and revision.
16. Least-autonomy test.
17. Budget rejection test.
18. Cache miss/hit test.

Create one command or clearly documented command sequence to run the complete acceptance suite.

### Release gate

Do not mark the project complete until every checkbox in `naukri_capstone_requirements.md` passes and all evidence is committed to the single repository.

---

## 13. Suggested repository layout

```text
naukri-domain-support-agent/
├── README.md
├── requirements.txt
├── .env.example
├── dataset.py
├── data/
├── knowledge_base/
├── docs/
│   ├── capstone_brief.md
│   ├── requirements_traceability.md
│   └── dependency_compatibility.md
├── specs/
│   ├── naukri_capstone_requirements.md
│   ├── naukri_capstone_architecture.md
│   └── naukri_capstone_implementation_plan.md
├── src/
│   ├── config/
│   ├── rag/
│   ├── tools/
│   ├── llm/
│   ├── crew/
│   ├── memory/
│   ├── guardrails/
│   ├── schemas/
│   ├── review/
│   ├── governance/
│   ├── cache/
│   ├── observability/
│   └── api/
├── evaluation/
│   └── reports/
├── transcripts/
└── tests/
```

The layout is a recommendation. Equivalent structure is acceptable if every required responsibility and evidence item exists.

---

## 14. Staged execution workflow

Use the following controlled workflow.

### Step 1 — Specification review and traceability

Inputs:
- `naukri_capstone_requirements.md`
- `naukri_capstone_architecture.md`
- `naukri_capstone_implementation_plan.md`

1. Read all three documents completely.
2. Treat the requirements file as authoritative.
3. Create a traceability matrix mapping every task and acceptance criterion to a module, test and evidence file.
4. Identify conflicts, ambiguous decisions and dependency risks.
5. Write no application code yet.
6. Record the repository tree and staged plan.

### Step 2 — Compatibility spikes

Implement only:
- dependency/version smoke tests;
- CrewAI custom `BaseLLM` mock prototype;
- tool schema dispatch tests including `rag_lookup`;
- AutoGen two-agent structured-message prototype;
- local SentenceTransformers/ChromaDB smoke test.

Do not proceed until all prototypes pass without API keys or network access.

### Step 3 onward — One phase at a time

For each phase in this plan:
1. Implement only that phase.
2. List the changed and created files.
3. Run the phase tests.
4. Map the test results to acceptance criteria.
5. Review the diff and evidence.
6. Confirm the phase tests pass before moving to the next one.

### Final step — Full acceptance audit

- Run all tests and demonstrations in offline `MOCK_LLM` mode.
- Inspect for any accidental network/API calls.
- Verify CrewAI telemetry is disabled before `.kickoff()`.
- Scan logs/transcripts for the raw fabricated phone test value.
- Verify every required category, status and KB topic.
- Verify all 15 evaluation rows have four scores and all four averages.
- Verify both AutoGen outcomes, budget rejection and cache hit.
- Produce a final pass/fail matrix without marking untested criteria as passed.

---

## 15. Definition of done

The project is ready for submission only when:
- the repository is public and contains one complete project;
- the README starts by identifying the Naukri.com (Recruitment & HR) track;
- dataset design choices are reproducible;
- all required categories, statuses and KB topics are present exactly as specified;
- all 16 tasks and acceptance criteria are implemented;
- all demonstration transcripts/reports are committed;
- all graded runs use `MOCK_LLM` with zero API keys and zero network access;
- CrewAI telemetry is disabled;
- no prohibited image, screenshot, PDF, slide, video or audio submission artefacts are included;
- one GitHub repository URL is ready to submit.

**Do not trade acceptance-criterion coverage for a superficially polished demo.**
