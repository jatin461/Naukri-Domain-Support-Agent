# Naukri.com Domain Support Agent — Requirements Specification

**Capstone track:** Recruitment & HR (Naukri.com)  
**Project:** Naukri.com Domain Support Agent using CrewAI, AutoGen, RAG and FastAPI  
**Specification status:** Approved requirements baseline  
**Source:** “Final Capstone — Naukri.com Domain Support Agent (`CrewAI`)” capstone brief  
**Target submission:** One public GitHub repository  
**Estimated duration:** 14 days  
**Total marks:** 100

---

## 1. Purpose and specification authority

Build one production-minded domain support agent for Naukri.com's employer-support scenario. It must answer hiring-policy questions from a locally authored knowledge base, retrieve actual job-application records from a deterministic dataset, maintain in-process conversation memory, enforce input and output guardrails, have its draft independently reviewed by an AutoGen team, and expose the solution through FastAPI.

This specification consolidates the capstone requirements into a single baseline. If any wording in this document is ambiguous, the capstone brief's wording governs.

### 1.1 Non-negotiable execution constraints

- All acceptance criteria must pass in deterministic `MOCK_LLM` mode.
- Graded demonstrations and transcripts must run with zero API keys and zero network access.
- `SentenceTransformers` embeddings and `ChromaDB` vector indexes must run locally.
- A real LLM may be optionally supported behind an environment-variable flag, but it is not required and must not be necessary for any acceptance criterion.
- Extend `crewai.llms.base_llm.BaseLLM` for CrewAI's mock LLM implementation. Do not rely on external interception of CrewAI calls.
- Disable CrewAI telemetry before any crew execution by setting `CREWAI_DISABLE_TELEMETRY=true` or `OTEL_SDK_DISABLED=true`. Confirm the setting in `README.md`.
- Use fabricated candidate/application examples only.
- Do not claim that candidate name, expected salary, or background-check result are reliably masked. Under this keyless mock-only design, these are explicitly out of scope for pattern-based masking. The fixed-format phone number in contact details must be masked.
- The only submission is one public GitHub repository containing code and text. No images, screenshots, uploaded diagrams, PDFs, slide decks, presentations, videos, or audio are required or accepted.

### 1.2 Submission and README requirements

The repository must contain all source code, dataset generator, generated dataset or reproducible generation process, knowledge-base documents, RAG implementation, CrewAI crew, AutoGen review stage, FastAPI application, governance controls, evaluation scripts, and demonstration transcripts.

At the top of `README.md`:
1. State that the completed track is **Naukri.com (Recruitment & HR)**.
2. State the exact dataset-design choices from Part 1 Task 1: random seed, category/status/priority-generation choices or weights, and expected-salary range.
3. Explain that all graded runs use `MOCK_LLM`, require no API keys, and make no network calls.
4. Confirm CrewAI telemetry is disabled before crew execution.
5. Include the measured similarity values and selected fallback threshold.
6. Include the escalation-score formula and escalation threshold.
7. Provide reproducible setup, run, and test instructions.

---

## 2. Product scope and user scenarios

### 2.1 Primary purpose

The agent supports recruiters and candidates by:
- Answering hiring-policy questions using only retrieved knowledge-base evidence.
- Looking up a specific job application's current status and expected salary from the generated dataset.
- Maintaining conversation context across turns within one process.
- Rejecting or safely handling prompt injection and unsupported questions.
- Masking the fixed-format phone-number PII field before model processing and logging.
- Producing structured responses.
- Submitting the CrewAI draft and original retrieved context to an independent AutoGen review team before returning the answer.
- Providing HTTP and WebSocket access through FastAPI.
- Recording structured logs and evaluation results.

### 2.2 Out of scope

- Making hiring, rejection, selection, ranking, or compensation decisions.
- Using real candidate data or real Naukri.com systems.
- Depending on paid APIs, paid accounts, or network access for graded demonstrations.
- Masking arbitrary free-text candidate names, expected salary values, or background-check results. These are acknowledged as out of scope for the specified keyless mock-only masker.
- Deliverables outside the single code-and-text GitHub repository.

---

## 3. Canonical domain vocabulary and knowledge base

### 3.1 Required job categories

Use every value below at least once. Additional values may be added, but these five values must remain unchanged and spelled exactly as shown:

- `Software Engineer`
- `Data Analyst`
- `Product Manager`
- `HR Executive`
- `Sales Associate`

Every required category must have at least 3 generated records.

This vocabulary is mandatory; the five category names must not be substituted or renamed.

### 3.2 Required application statuses

Use every value below at least once. Additional values may be added:

- `Applied`
- `Screening`
- `Interview Scheduled`
- `Offered`
- `Rejected`

Every required status must have at least 1 generated record.

### 3.3 Required knowledge-base documents

Author at least 12 distinct documents, each containing 2–5 sentences, written specifically for this project. Cover every topic below:

1. Job-application eligibility criteria.
2. Interview-scheduling process.
3. Offer-negotiation policy.
4. Background-verification process.
5. Notice-period policy.
6. Referral-bonus policy.
7. Internal-transfer eligibility.
8. Probation-period policy.
9. Remote-work eligibility.
10. Diversity-hiring guidelines.
11. Exit-interview process.
12. Applicant-data-retention policy.

Additional documents may be added, but all 12 required topics must be present exactly as listed.

Each document must have a stable document identifier/title so retrieved chunks can be mapped back to their parent document for evaluation.

---

## 4. Part 1 — Dataset Design and RAG Core (30 marks)

### 4.1 Task 1: Deterministic job-application dataset

Implement `dataset.py` with a seeded, deterministic Python generator producing a `JOB_APPLICATIONS` list of at least 40 records.

Each record must contain:
- `record_id`
- `category`
- `status`
- `expected_salary_inr`
- `days_since_created` — integer from 0 through 30 inclusive
- `flagged_priority_review` — boolean

Dataset design requirements:
- Use all five required categories, with at least 3 records per category.
- Use all five required statuses, with at least 1 record per status.
- Generate `flagged_priority_review=True` for 10%–30% of records.
- If the initial seeded random generation does not meet the priority percentage, adjust the seed or generation weights and regenerate. Never hand-edit individual records to force the percentage.
- Choose a realistic expected-salary range and explain the reasoning in one sentence.
- The exact seed, generation weights/choices, and salary range must be documented at the top of `README.md` so the dataset can be reproduced.
- The seed value is a design choice. Any seed is acceptable (for example `42`) provided the resulting generated data satisfies every required distribution constraint.

Print and report:
- Count per category.
- Count per status.
- Percentage of records with `flagged_priority_review=True`.

Include a validation routine or test that asserts all structural and distribution constraints.

### 4.2 Task 2: Knowledge-base authoring

Create at least the 12 required policy documents listed in Section 3.3. Each document must contain 2–5 sentences and be written specifically for this project. Ensure all required topics are covered and represented in the retrieval corpus.

### 4.3 Task 3: Two chunking strategies and two ChromaDB collections

Implement both strategies over the same knowledge-base documents:

**Strategy A — fixed-size with overlap**
- Split document text into fixed-size chunks.
- Apply a non-zero overlap.
- Preserve parent-document metadata and chunk identifiers.

**Strategy B — sentence-based**
- Split text at sentence boundaries into sentence-based chunks.
- Preserve parent-document metadata and chunk identifiers.

For both strategies:
- Embed every chunk using a free, locally executed `SentenceTransformers` model.
- Index each strategy in its own separate `ChromaDB` collection.
- Use `collection.upsert()` for indexing.
- Ensure the two collections are independently queryable and return sensible retrieval results.

### 4.4 Task 4: Grounded generation and empirical fallback calibration

Given a query:
1. Retrieve the top-k chunks from the selected collection.
2. Generate an answer using only the retrieved context.
3. Under `MOCK_LLM`, use retrieval similarity as the groundedness/fallback signal; do not claim that a real model judged semantic groundedness.
4. Return an explicit “I don't know”/unsupported-question fallback when the calibrated threshold is not met.

Calibrate the threshold empirically:
- Measure top-1 cosine similarity for at least 3 in-scope queries.
- Measure top-1 cosine similarity for at least 2 deliberately out-of-scope queries.
- Record the actual measurements.
- Select a threshold between the clusters observed in those measurements.
- Do not use an untested preset such as 0.5, 0.6, or 0.7.
- Document the measurements, selected threshold, and rationale in `README.md`.

Demonstrate:
- At least 5 real in-scope queries that receive context-grounded answers.
- At least 1 deliberately out-of-scope query that triggers the fallback.

### 4.5 Task 5: Compare chunking strategies using document-level precision and recall

Use the same minimum 5 in-scope queries used for Task 4.

For each query and each collection:
- Define the relevant/expected parent document(s) for the query.
- Retrieve chunks and map each chunk to its parent document.
- Deduplicate parent-document IDs before scoring.
- Calculate document-level precision and recall separately for the fixed-size-with-overlap collection and the sentence-based collection.
- Show the per-query arithmetic for both strategies; do not provide only aggregate scores.
- Provide 2–3 sentences recommending which strategy to deploy, citing the actual results from both sets of measurements.

The recommended collection becomes the fixed RAG input to Part 2.

---

## 5. Part 2 — CrewAI orchestration, tools, memory and guardrails (30 marks)

Use the recommended Part 1 RAG collection as a fixed input to the orchestration layer.

### 5.1 Task 6: Application-status lookup tool and escalation score

Implement:

`check_job_application_status(record_id: str) -> dict`

The function must look up the generated dataset and return:
- Application `status`.
- `expected_salary_inr`.
- A designed `escalation_score` in the inclusive range `[0, 1]`.

The score must combine:
- `flagged_priority_review`.
- A normalized recency signal derived from `days_since_created`.

It must be a genuine designed score, not a bare boolean OR. State the exact formula and explain the contribution of each component.

Also:
- State the escalation threshold above which escalation is recommended.
- Justify the threshold using the actual generated dataset distribution. For example, relate the recency component or chosen cutoff to the observed 80th percentile of `days_since_created`.
- Do not assert a percentile relationship unless it is calculated from the generated dataset.
- Demonstrate correct lookup behaviour, including an unknown `record_id` case.

A proposed design that may be used if it is validated and justified against the generated dataset:
`escalation_score = 0.60 * flagged_priority_review + 0.40 * (1 - days_since_created / 30)`.
The exact formula is a design choice; the requirement is that the score combines the priority flag and a normalized recency signal.

### 5.2 Task 7: CrewAI crew

Build a CrewAI crew with at least 3 distinct agents:

1. **Retrieval Agent**
   - Equipped with the RAG retrieval tool from Part 1.
   - Retrieves policy evidence for policy questions.

2. **Lookup Agent**
   - Equipped with `check_job_application_status`.
   - Retrieves the requested application status and associated fields.

3. **Response Composer**
   - Combines the relevant retrieval and/or lookup outputs into one final draft response.

Requirements:
- Run the crew using `.kickoff()`.
- Demonstrate the RAG tool actually being invoked.
- Demonstrate the lookup tool actually being invoked.
- Use different sample queries to demonstrate the two tools.
- Tool invocation evidence must be included in transcripts/logs or tests.

#### Required CrewAI `MOCK_LLM` implementation

- Extend `crewai.llms.base_llm.BaseLLM`, CrewAI's documented extension point for a non-LiteLLM LLM.
- Do not attempt to satisfy this requirement by intercepting model calls externally.
- Parse the model's generated text, not the system-prompt template. CrewAI's built-in ReAct system prompt includes the literal example text `Observation: the result of the action`; searching the whole conversation for `Observation:` can match the template before a tool has run and silently return placeholder output.
- Dispatch tool arguments using the tool's declared argument schema, not by matching substrings in the tool name. A tool named `rag_lookup` must not be misclassified merely because its name contains `lookup`.
- Test the mock LLM with both required tools and include evidence that the correct tool and arguments were selected.

### 5.3 CrewAI telemetry and offline execution

Before running the crew, set one of:
- `CREWAI_DISABLE_TELEMETRY=true`, or
- `OTEL_SDK_DISABLED=true`.

Set it as an environment variable before crew execution, not after `.kickoff()`.

Confirm the setting in `README.md`. The graded `MOCK_LLM` execution must not make outbound network calls.

### 5.4 Task 8: In-process session memory

Use LangChain session-based memory, for example:
- `InMemoryChatMessageHistory`
- `RunnableWithMessageHistory`

Requirements:
- Preserve conversation history across multiple turns within one process run.
- Demonstrate a multi-turn exchange in one transcript where context from an earlier turn is correctly used in a later turn.
- Demonstrate a separate fresh conversation/session transcript where the previous session's state is absent/reset.
- Memory only needs to persist in process; it does not need to survive application restart.
- The `RunnableWithMessageHistory` deprecation warning pointing to LangGraph persistence is expected and does not need to be silenced.

### 5.5 Task 9: Structured crew response

Define a Pydantic `BaseModel` for the crew's response format.

Requirements:
- Every crew response must conform to the declared schema.
- Validate every response against the schema in code.
- Include validation evidence in tests or transcripts.
- The schema applies to the crew's response. Each individual agent execution loop is not required to use its own separate schema; the required deliverable is that every crew response conforms to and is validated against the declared response schema.

### 5.6 Task 10: Guardrails

Implement and demonstrate all of the following:

**Input-side PII masking**
- Identify the fixed-format phone number field inside contact details.
- Mask it before the input reaches the model.
- Demonstrate the masking guardrail firing with a deliberate test case.
- Use fabricated examples only.
- Candidate name, expected salary, and background-check results are explicitly out of scope for masking in this keyless `MOCK_LLM` design; do not claim they are masked.

**Input-side prompt-injection detection**
- Detect and block or safely neutralize a deliberate prompt-injection test case.
- Demonstrate the guardrail firing.
- Do not rely on a claim that a short regex dictionary can detect every possible paraphrase. State the scope and limitations of the deterministic mock-mode detector.

**Output-side groundedness check**
- Check whether the draft answer is supported by retrieved context.
- Refuse or fall back when retrieved context does not support the requested answer.
- Demonstrate the groundedness guardrail firing with a deliberate unsupported/ungrounded test case.

---

## 6. Part 3 — Evaluation, observability and FastAPI deployment (20 marks)

Use the Part 2 crew as a fixed input.

### 6.1 Task 11: FastAPI deployment

Expose at least two HTTP endpoints using Pydantic request/response models, plus one WebSocket endpoint.

Required examples:
- `POST /ask` — process a standard support query.
- `POST /add-document` — add a document to the local vector engine.
- `@app.websocket("/chat")` — real-time multi-turn chat.

Requirements:
- The HTTP endpoints must work.
- Use Pydantic models for request/response payloads where applicable.
- The WebSocket must support multiple turns.
- Catch `WebSocketDisconnect` and handle a client disconnecting mid-conversation gracefully.
- A disconnect must not crash or stop the server from serving other clients.
- The WebSocket endpoint must handle concurrent connections gracefully using standard async Python patterns: an `async def` handler, with blocking work (for example CrewAI `.kickoff()`) kept off the event loop, such as in a thread pool, so a slow or disconnected client does not block the others.
- Multi-turn memory state must remain isolated per WebSocket connection/session ID: a turn on one connection or session must never read or modify the history of another. A connection that supplies no session ID is given a unique one; connections that deliberately supply the same session ID share that session's history.
- Include a test in which at least two WebSocket clients hold interleaved multi-turn conversations at the same time and each receives answers that use only its own session's context.
- Include runnable tests or demonstration evidence for the endpoints and disconnect handling.

### 6.2 Task 12: Structured JSON-Lines logging and PII safety

Log every request as one JSON-Lines (`.jsonl`) entry.

Each entry must include:
- A unique trace ID.
- Timestamp.
- Request timing/latency information.
- Relevant request/response metadata suitable for operational troubleshooting.

PII requirements:
- The logged request text must not contain the raw value of any fixed-format phone number that the input guardrail masks.
- Apply the same phone-number masking to log content as to model input.
- Mask before writing to disk; do not write raw PII and attempt to redact it afterward.
- Demonstrate with a test that the raw fabricated phone number is absent from the log file and the masked representation is present.
- Do not claim masking of the explicitly out-of-scope free-text fields.

### 6.3 Task 13: 15-query evaluation suite

Build a test set of exactly 15 queries and evaluate under `MOCK_LLM`.

Coverage:
- At least one query touching each of the 12 required knowledge-base topics.
- At least 2 deliberately out-of-scope or edge-case queries.
- The 15-query set must therefore cover all 12 policy topics and at least 2 out-of-scope/edge cases. One additional query may be used for another relevant scenario.

Use an LLM-as-judge prompt/interface that runs deterministically under `MOCK_LLM`. The judge must not require an actual paid or network LLM call.

For every one of the 15 queries, report all four scores:
1. Accuracy.
2. Grounding.
3. Completeness.
4. Safety.

Also report the average of each of the four scores across all 15 queries.

Requirements:
- Include the query, individual scores, and enough output/evidence to understand the evaluation.
- Make the scoring rubric and score scale explicit.
- Report all 4 scores for every query; category-level or aggregate-only reporting is insufficient.
- Report all four averages.
- Preserve the `MOCK_LLM`-only, zero-network execution requirement.

---

## 7. Part 4 — Resilience and governance (20 marks)

Use the Part 2 crew as a fixed input.

### 7.1 Task 14: Independent AutoGen review stage

Add an AutoGen review stage after the CrewAI Response Composer creates its draft.

Build a 2-agent AutoGen Round Robin Group Chat team:
- `Policy-Compliance-Reviewer`
- `Final-Editor`

Required orchestration:
- Use `RoundRobinGroupChat`.
- Bound the team with `max_turns=2`.
- If using `MaxMessageTermination` instead, account for the initiating task message as message 1; `MaxMessageTermination(3)` is required to allow both agents to speak.
- Give the Final-Editor a Pydantic structured output using `output_content_type=<YourVerdictModel>` on the agent.
- Construct the team with `custom_message_types=[StructuredMessage[YourVerdictModel]]`; this registration is required for the structured message type.
- The team receives both the CrewAI Composer's draft answer and the original retrieved context.
- The team either approves the draft unchanged or revises it based on policy compliance and evidence.

Define a Pydantic verdict model with exactly these required fields:
- `approved: bool`
- `final_answer: str`
- `reason: str`

Demonstrate at least 2 sample queries:
1. A case where the review stage approves the draft unchanged.
2. A case where the review stage revises the draft, such as a deliberately injected ungrounded claim being detected and removed.

Include structured verdicts and evidence for both cases.

### 7.2 Task 15: Four-layer AI governance

Apply the required governance controls.

#### Application layer — least autonomy and risk classification

Least autonomy:
- Only the Lookup Agent may call `check_job_application_status`.
- Do not wire the lookup tool to the Retrieval Agent, Response Composer, or any other agent.
- Demonstrate the restriction, for example through tool wiring inspection or a test showing that another agent cannot invoke the lookup tool.
- Include a one-paragraph explanation of how the restriction is enforced.

Risk classification:
- Classify the system using the specified Low/Medium/High scheme:
  - Low: summarization/transcription.
  - Medium: code generation/customer-support tickets.
  - High: medical data, hiring decisions, financial data.
- State one explicit classification and justify it in a paragraph based on what this system actually does and does not do.
- Do not invent capabilities such as candidate salary ranking or automated hiring decisions. If the agent is restricted to policy support and status lookup, distinguish that from a system that makes hiring decisions.

#### Runtime layer — per-request token/cost budget

- Enforce a per-request token/cost budget cap.
- Reject a deliberately oversized request before it silently exceeds the budget.
- Demonstrate the rejection in a test or transcript.
- Explain how the budget is calculated in deterministic `MOCK_LLM` mode (for example, a deterministic token estimate and configured cap). Do not imply that mock execution incurs an actual provider charge.

### 7.3 Task 16: Response caching

Implement an in-memory cache for the grounded-generation step.

Requirements:
- Cache key is normalized query text. Normalization must strip leading and trailing whitespace and punctuation and convert the text to lower case, so that trivially different forms of the same query hit the same entry. Collapsing internal runs of whitespace to a single space is part of the same rule, and punctuation inside the query (for example the hyphen in a record ID) is preserved.
- The normalization is purely lexical. Do not claim that it matches paraphrases or semantically equivalent queries; the cache-hit benefit applies to variants that differ only in case, surrounding whitespace or surrounding punctuation.
- A query that is empty after normalization is a malformed request and is rejected by input validation before it reaches the cache.
- A repeated identical/normalized query must produce a cache hit.
- The cache hit must avoid a redundant LLM/tool call for the grounded-generation step.
- Demonstrate before/after evidence using a call counter or timing comparison.
- Ensure cache behaviour does not bypass input guardrails, per-request budget checks, session handling, or required review/response validation. Cache only the grounded-generation result as specified, unless additional caching is separately justified and does not weaken controls.

---

## 8. Cross-cutting non-functional requirements

### 8.1 Reproducibility
- Seeded dataset generation.
- Deterministic mock responses.
- Documented dependency versions or a reproducible dependency specification.
- Clear commands for setup, indexing, running the API, and executing tests/evaluation.

### 8.2 Privacy and security
- Use fabricated records.
- Mask fixed-format phone numbers before model input and disk logging.
- Detect prompt injection and refuse/neutralize unsafe requests.
- Enforce least privilege for agent tools.
- Do not log secrets or API keys.
- No network calls in graded mock-mode execution.

### 8.3 Reliability
- Handle missing application IDs.
- Handle unsupported policy questions.
- Handle empty or malformed requests.
- Handle WebSocket disconnects without taking down the service.
- Validate structured outputs before returning them.
- Ensure AutoGen review is applied after CrewAI drafting and before user delivery.

### 8.4 Observability
- JSON-Lines request logging with trace ID and timing.
- Tool invocation evidence.
- Cache hit/miss evidence.
- Evaluation report with per-query scores and averages.
- Transcripts for memory, guardrails, AutoGen approval/revision, and budget rejection.

---

## 9. Required repository deliverables

The final repository must contain, at minimum, the following logical artefacts. Exact module/file names may be chosen by the implementation, but all responsibilities must be present.

```text
naukri-domain-support-agent/
├── README.md
├── requirements.txt                 # or another reproducible dependency manifest
├── .env.example                     # no real secrets
├── dataset.py
├── data/
│   └── generated_job_applications.json  # optional generated snapshot
├── knowledge_base/
│   ├── job_application_eligibility.md
│   ├── interview_scheduling.md
│   ├── offer_negotiation.md
│   ├── background_verification.md
│   ├── notice_period.md
│   ├── referral_bonus.md
│   ├── internal_transfer.md
│   ├── probation_period.md
│   ├── remote_work.md
│   ├── diversity_hiring.md
│   ├── exit_interview.md
│   └── applicant_data_retention.md
├── src/
│   ├── config/
│   ├── rag/
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── retrieval.py
│   │   └── grounded_generation.py
│   ├── tools/
│   │   ├── rag_tool.py
│   │   └── application_lookup.py
│   ├── llm/
│   │   └── mock_crewai_llm.py
│   ├── crew/
│   │   ├── agents.py
│   │   ├── tasks.py
│   │   └── crew.py
│   ├── memory/
│   ├── guardrails/
│   │   ├── input_guardrails.py
│   │   └── output_guardrails.py
│   ├── schemas/
│   ├── review/
│   │   └── autogen_review.py
│   ├── governance/
│   │   ├── autonomy.py
│   │   ├── budget.py
│   │   └── risk_assessment.md
│   ├── cache/
│   ├── observability/
│   │   └── logging_config.py
│   └── api/
│       └── main.py
├── evaluation/
│   ├── evaluation_dataset.py
│   ├── judge.py
│   └── reports/
├── transcripts/
│   ├── rag_demos.md
│   ├── crew_tool_invocations.md
│   ├── memory_sessions.md
│   ├── guardrail_demos.md
│   ├── autogen_review_demos.md
│   ├── governance_budget_demo.md
│   └── cache_demo.md
└── tests/
    ├── test_dataset.py
    ├── test_rag.py
    ├── test_tools_and_crew.py
    ├── test_memory.py
    ├── test_guardrails.py
    ├── test_api.py
    ├── test_evaluation.py
    ├── test_review.py
    └── test_governance_cache.py
```

This is a recommended logical structure, not a requirement to use these exact filenames. Do not omit any required capability merely because a different file layout is chosen.

---

## 10. Acceptance criteria and traceability checklist

The project is complete only when the following are demonstrably satisfied.

### Part 1 — 30 marks

- [ ] `dataset.py` deterministically generates at least 40 records.
- [ ] Every required category appears at least 3 times.
- [ ] Every required status appears at least once.
- [ ] Priority-flagged percentage is between 10% and 30%, without hand-editing records.
- [ ] Required dataset fields and salary rationale are documented.
- [ ] At least 12 project-authored knowledge-base documents exist, each 2–5 sentences.
- [ ] All 12 exact required policy topics are covered.
- [ ] Fixed-size-with-overlap chunking is implemented.
- [ ] Sentence-based chunking is implemented.
- [ ] Both chunk sets are embedded with local SentenceTransformers.
- [ ] Each strategy has its own ChromaDB collection and uses `collection.upsert()`.
- [ ] Similarity threshold is empirically calibrated using at least 3 in-scope and 2 out-of-scope queries.
- [ ] At least 5 in-scope RAG demonstrations return grounded answers.
- [ ] At least 1 out-of-scope query triggers the fallback.
- [ ] Document-level precision and recall are calculated for both collections on the same minimum 5 queries.
- [ ] Per-query arithmetic is visible for both strategies.
- [ ] A 2–3 sentence, numbers-cited chunking recommendation is included.

### Part 2 — 30 marks

- [ ] `check_job_application_status(record_id: str) -> dict` returns status, expected salary and a designed escalation score in `[0,1]`.
- [ ] Score combines priority flag and normalized recency.
- [ ] Formula and dataset-justified escalation threshold are documented.
- [ ] CrewAI crew has at least 3 agents: Retrieval, Lookup and Composer.
- [ ] Crew executes through `.kickoff()`.
- [ ] RAG tool invocation is demonstrated.
- [ ] Lookup tool invocation is demonstrated on a different sample query.
- [ ] CrewAI mock LLM extends `crewai.llms.base_llm.BaseLLM`.
- [ ] ReAct `Observation:` parsing pitfall is avoided by parsing generated model text rather than the system prompt.
- [ ] Tool dispatch uses declared argument schema rather than tool-name substring matching.
- [ ] CrewAI telemetry is disabled before execution and documented.
- [ ] Multi-turn in-process memory is demonstrated.
- [ ] A separate fresh session demonstrates reset/absence of prior state.
- [ ] Every crew response validates against a Pydantic schema.
- [ ] Fixed-format phone PII masking is demonstrated.
- [ ] Prompt-injection guardrail is demonstrated.
- [ ] Output groundedness refusal is demonstrated.

### Part 3 — 20 marks

- [ ] At least two working FastAPI HTTP endpoints exist.
- [ ] One working FastAPI WebSocket endpoint supports multi-turn chat.
- [ ] WebSocket disconnect is caught and does not crash the server.
- [ ] Applicable HTTP request/response payloads use Pydantic models.
- [ ] Every request creates one JSON-Lines log entry with trace ID and timing.
- [ ] Raw fixed-format phone PII never reaches the log file.
- [ ] A 15-query evaluation set covers all 12 KB topics and at least 2 out-of-scope/edge cases.
- [ ] Accuracy, Grounding, Completeness and Safety are reported for every query.
- [ ] The average of each of the four scores is reported across all 15 queries.
- [ ] Evaluation and judge run under `MOCK_LLM`.
- [ ] WebSocket handles concurrent connections with async patterns, and multi-turn memory is isolated per connection/session ID (shown by a test with interleaved concurrent clients).

### Part 4 — 20 marks

- [ ] AutoGen review team has exactly the required two reviewer/editor roles.
- [ ] Round-robin orchestration is bounded with `max_turns=2`, or the correctly configured alternative termination condition.
- [ ] Final-Editor uses `output_content_type=<YourVerdictModel>`.
- [ ] Team registers `custom_message_types=[StructuredMessage[YourVerdictModel]]`.
- [ ] Review receives CrewAI draft and original retrieved context.
- [ ] Pydantic verdict includes `approved`, `final_answer`, and `reason`.
- [ ] One demonstration approves unchanged.
- [ ] One demonstration revises an intentionally ungrounded draft.
- [ ] Only Lookup Agent is wired to `check_job_application_status`.
- [ ] Least-autonomy restriction is demonstrated and explained.
- [ ] Risk level is classified using the supplied Low/Medium/High scheme and justified without invented capabilities.
- [ ] Per-request token/cost budget cap is enforced.
- [ ] Oversized request is rejected and the rejection is demonstrated.
- [ ] Grounded-generation cache is keyed by normalized query text.
- [ ] Repeated query demonstrates a cache hit and avoided redundant grounded-generation LLM/tool call with evidence.
- [ ] Cache-key normalization strips leading/trailing whitespace and punctuation and lower-cases, variants of one query share an entry, and no semantic-match claim is made.

### Submission

- [ ] One public GitHub repository link contains all code, knowledge-base documents, dataset, tests, reports and transcripts.
- [ ] README begins by identifying the Naukri.com (Recruitment & HR) track.
- [ ] README documents exact dataset-generation choices.
- [ ] README confirms zero API keys and zero network access for graded `MOCK_LLM` runs.
- [ ] README confirms CrewAI telemetry is disabled.
- [ ] README contains measured similarity values and chosen threshold.
- [ ] No prohibited non-code/text deliverables are used as submission artefacts.

---

## 11. Implementation stages

Use this specification as the approved requirements baseline. Development should proceed in controlled stages:

1. Implement dataset generation and validation.
2. Author and validate all required KB documents.
3. Implement both chunking strategies, local embeddings, separate ChromaDB collections and retrieval evaluation.
4. Calibrate and document the grounded-generation fallback.
5. Implement the application lookup tool and escalation calculation.
6. Implement and test CrewAI's custom mock LLM before building the full crew.
7. Add the crew, session memory, structured response and guardrails.
8. Add FastAPI, WebSocket handling and PII-safe structured logging.
9. Add the 15-query evaluation suite.
10. Add the AutoGen review stage and structured verdict.
11. Add least-autonomy enforcement, risk classification, runtime budget and cache.
12. Run the full acceptance checklist in offline `MOCK_LLM` mode and capture repository transcripts/reports.

For every stage:
- Write tests before marking the stage complete.
- Do not silently invent business rules when the source requirements are silent; document implementation choices.
- Keep the required vocabulary and policy topics unchanged.
- Do not mark an acceptance criterion complete without executable evidence.
- Do not proceed by replacing a missing requirement with a superficially similar feature.

**Final release gate:** All acceptance criteria in Section 10 must pass in `MOCK_LLM` mode with zero API keys and zero network access before the repository is submitted.
