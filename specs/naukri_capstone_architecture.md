# Naukri.com Domain Support Agent — Architecture & Component Flow

**Project:** Naukri.com Domain Support Agent  
**Track:** Recruitment & HR  
**Architecture baseline:** `naukri_capstone_requirements.md`  
**Execution mode:** Deterministic, local, offline `MOCK_LLM` for all graded demonstrations  
**Status:** Development architecture specification

> **Submission constraint:** The capstone brief explicitly says that no diagrams are required or accepted as submission artefacts. This document therefore describes the architecture using text-only component maps and numbered flows. Do not add image diagrams, screenshots, slide decks, or exported diagrams to the submitted repository. This text document is a development aid.

---

## 1. Architecture objectives

The solution must provide one locally executable support-agent application that:

1. Answers policy questions only from the authored knowledge base.
2. Looks up application records from a deterministic fabricated dataset.
3. Uses two RAG chunking strategies for comparison, then uses the empirically selected collection in the agent crew.
4. Maintains session history within a running process.
5. masks fixed-format phone-number PII before model input and before disk logging.
6. Detects prompt-injection attempts and unsupported questions.
7. Uses CrewAI for the retrieval, lookup and response-composition workflow.
8. Uses an independent two-agent AutoGen review stage before returning a draft.
9. Validates structured outputs with Pydantic.
10. Exposes HTTP and WebSocket interfaces through FastAPI.
11. Enforces per-request budget, least autonomy, structured logging and grounded-generation caching.
12. Demonstrates every required capability under zero-key, zero-network `MOCK_LLM` execution.

## 2. Architecture principles and boundaries

### 2.1 Single source of truth

The requirements document is the functional source of truth. Do not alter the required categories, statuses or policy topics, and do not invent acceptance criteria or business behaviours.

### 2.2 Separation of responsibilities

- Dataset module owns fabricated job-application records and deterministic generation.
- Knowledge-base module owns authored policy text and document metadata.
- RAG module owns chunking, embeddings, vector collections, retrieval, similarity calibration and grounded generation.
- Tool module exposes narrow RAG and application-lookup capabilities.
- CrewAI module owns agent orchestration and draft composition.
- Guardrail module owns input sanitization and output groundedness validation.
- Memory module owns session-specific, in-process conversation history.
- AutoGen module owns independent review of the draft against original retrieved context.
- Governance module owns least-autonomy checks, risk classification documentation and request budget enforcement.
- Cache module owns normalized-query caching for grounded generation only.
- API module owns transport, request validation, session selection and response delivery.
- Observability module owns trace IDs, timings and PII-safe JSONL logging.
- Evaluation module owns the 15-query evaluation set, deterministic judge and score reports.

### 2.3 Offline runtime boundary

All graded execution must be local and offline:
- No hosted LLM endpoint.
- No embedding API.
- No remote vector database.
- No external telemetry.
- No network-dependent judge.
- No API key requirement.

The embedding model must be present in the local model cache before the graded offline run. Model installation/download, if needed, is a preparation activity; the actual graded run must not fetch model files.

### 2.4 No real candidate data

Use fabricated records only. The fixed-format phone-number field is the only PII field required to be masked. Candidate name, expected salary and background-check results are explicitly out of scope for the keyless mock-only masker.

---

## 3. Logical component map

The application is divided into the following logical areas.

### 3.1 Data and knowledge layer

**`dataset.py`**
- Generates at least 40 deterministic application records.
- Uses exactly the required category and status vocabularies.
- Validates category counts, status coverage, age range, salary field and priority-flag percentage.
- Prints category/status counts and priority percentage.
- Exposes a lookup-friendly `JOB_APPLICATIONS` collection or an indexed mapping.

**`knowledge_base/`**
- Contains at least 12 policy documents, each 2–5 sentences.
- Uses the exact 12 policy topics from the capstone.
- Gives each document a stable ID/title.

**`rag/`**
- Loads the authored documents.
- Creates fixed-size-with-overlap chunks and sentence-based chunks.
- Stores `document_id`, `chunk_id`, source/title and strategy metadata.
- Embeds all chunks with a local SentenceTransformers model.
- Upserts each strategy into its own ChromaDB collection.
- Measures top-1 cosine similarity for the calibration query set.
- Performs document-level precision/recall evaluation.
- Provides the selected collection as the Part 2 RAG source.

### 3.2 Agent and orchestration layer

**RAG retrieval tool**
- Accepts a user policy question.
- Calls the grounded-generation service.
- Returns answer, retrieved context, similarity evidence, parent document IDs and fallback status.
- Uses the normalized-query cache at the grounded-generation boundary.

**Application lookup tool**
- Accepts `record_id`.
- Reads only from the fabricated dataset.
- Returns status, expected salary and calculated escalation score.
- Is wired only to the Lookup Agent.

**CrewAI crew**
- Retrieval Agent: has only the RAG tool.
- Lookup Agent: has only the application lookup tool.
- Response Composer: has no application lookup tool; synthesizes the evidence and tool results.
- Uses a custom `BaseLLM` subclass for deterministic `MOCK_LLM`.
- Runs through `.kickoff()`.
- Validates the final crew response using a Pydantic model.

**Session memory**
- Uses `InMemoryChatMessageHistory` and `RunnableWithMessageHistory`, or the equivalent required LangChain session wrapper.
- Maintains separate histories by `session_id`.
- Resets naturally when a new session ID is used or the process restarts.

### 3.3 Guardrail and governance layer

**Input guardrails**
1. Normalize/validate incoming request fields.
2. Mask the fixed-format phone number.
3. Detect known prompt-injection patterns and reject or neutralize the request according to a documented deterministic policy.
4. Ensure the masked request is the version passed to the model and logger.

**Output guardrails**
- Verify that the generated answer is supported by retrieved context.
- Refuse or use the specified fallback when context is insufficient.
- Do not treat the mock judge as a real semantic model; similarity and deterministic checks are the mock-mode signals.

**Governance**
- Lookup tool is available only to the Lookup Agent.
- Per-request token estimate and configured budget are checked before crew execution.
- An oversized request is rejected with a structured error.
- Risk classification and justification are documented.

**Cache**
- Key is normalized query text.
- Stores grounded-generation result for that query.
- Cache hit avoids repeating the grounded-generation model/tool work.
- Cache does not bypass input guardrails, budget checks, session handling, final response validation or AutoGen review.

### 3.4 Independent review layer

**AutoGen review team**
- Policy-Compliance-Reviewer agent.
- Final-Editor agent.
- Round-robin team bounded by `max_turns=2`.
- Final-Editor uses the Pydantic verdict model via `output_content_type`.
- Team registers `StructuredMessage[YourVerdictModel]` in `custom_message_types`.
- Receives both the CrewAI draft and original retrieved context.
- Returns an approved unchanged answer or a revised answer with a reason.

### 3.5 API and observability layer

**FastAPI**
- `POST /ask`
- `POST /add-document`
- `WebSocket /chat` (async handler; blocking pipeline work runs in a thread pool; per-connection/session isolation)

**Logging**
- One JSONL event per request.
- Trace ID, timestamp, duration and safe request metadata.
- Phone masking occurs before logging.
- Raw phone value must never be written to disk.

---

## 4. End-to-end request flows

### 4.1 Policy-question flow

1. Client sends a policy question to `POST /ask` or the WebSocket.
2. FastAPI validates the request with a Pydantic request model.
3. API creates or accepts a trace ID and identifies the session.
4. Input guardrails inspect the request:
   - fixed-format phone values are masked;
   - prompt injection is rejected or neutralized.
5. Runtime budget guard estimates request tokens and rejects oversized input before crew execution.
6. Session memory retrieves the conversation history for the supplied session ID.
7. CrewAI `.kickoff()` starts the agent workflow.
8. Retrieval Agent invokes the RAG tool.
9. RAG service normalizes the query and checks the grounded-generation cache.
10. On a cache miss, RAG retrieves top-k chunks from the recommended ChromaDB collection, evaluates similarity against the empirically calibrated threshold, and generates a context-only mock answer.
11. On a cache hit, the cached grounded-generation result is returned without repeating the grounded-generation call.
12. Response Composer produces a structured draft.
13. Pydantic validates the crew response.
14. Output groundedness guard checks the draft against the retrieved context.
15. AutoGen receives the draft plus original retrieved context.
16. The reviewer/editor team returns a structured verdict:
    - approved unchanged, or
    - revised with a reason.
17. The final response is validated and returned to the client.
18. Session history is updated.
19. A single JSONL log entry is written using the masked request text and trace/timing metadata.

### 4.2 Application-status lookup flow

1. Client submits a request containing a `record_id`.
2. FastAPI validates the request and runs input guardrails and budget checks.
3. CrewAI `.kickoff()` routes the task to the Lookup Agent.
4. Only the Lookup Agent has `check_job_application_status`.
5. Lookup tool reads the fabricated dataset and returns status, expected salary and escalation score.
6. Unknown record IDs return a controlled not-found result; they must not produce invented records.
7. Response Composer turns the lookup result into a structured draft.
8. Pydantic validation runs.
9. AutoGen reviews the draft against the lookup result/context and returns a structured verdict.
10. API returns the approved/revised response and writes a PII-safe JSONL log.

### 4.3 Add-document flow

1. Client submits a document payload to `POST /add-document`.
2. FastAPI validates the payload with a Pydantic model.
3. The document is assigned a stable ID or validated caller-supplied ID.
4. The document is chunked using the selected production strategy.
5. Chunks are embedded locally.
6. Chunks are upserted into the selected ChromaDB collection with parent-document metadata.
7. The endpoint returns a structured result including document ID and indexed chunk count.
8. The operation is logged with trace ID and duration.
9. The endpoint must not silently change the 12 required policy documents; additional documents are additive.

### 4.4 WebSocket multi-turn flow

1. Client connects to `/chat`.
2. Server accepts the connection and associates a session ID.
3. Each received message goes through the same validation, PII masking, injection detection and budget checks as `/ask`.
4. Each turn uses the session's in-process memory.
5. The response follows the same CrewAI → guardrail → AutoGen review → final validation path.
6. Server sends the structured response.
7. On `WebSocketDisconnect`, the handler exits cleanly and releases per-connection resources.
8. The exception must not terminate the FastAPI server or affect other client sessions.
9. Concurrent connections are served by an `async def` handler with blocking pipeline work (CrewAI, AutoGen) kept off the event loop, for example via a thread pool, so one slow or disconnected client does not block the others.
10. Session memory is keyed by session ID, and each connection without a supplied session ID receives a unique one, so history from one connection/session is never read or modified by another. Shared in-process state (session store, cache, log writer) must be safe under concurrent access.

---

## 5. Component contracts

The implementation should keep the following logical contracts. Exact Python signatures may be refined, but the behaviour must remain.

| Component | Input | Output / responsibility |
|---|---|---|
| Dataset generator | Seed and generation configuration | Deterministic records and distribution report |
| Dataset validator | Generated records | Pass/fail assertions and counts |
| KB loader | KB document paths | Stable document objects with IDs and text |
| Fixed chunker | Document objects, chunk size, overlap | Chunks with parent IDs |
| Sentence chunker | Document objects | Sentence-based chunks with parent IDs |
| Embedding adapter | Chunk text | Local embedding vectors |
| Chroma indexer | Chunks, embeddings, collection name | Upserted vector records |
| Retriever | Query, top-k, collection | Ranked chunks and similarity scores |
| Grounded generator | Query and retrieved context | Context-only answer or fallback |
| RAG tool | Query | Tool result including evidence and fallback status |
| Application lookup | Record ID | Status, expected salary, escalation score or not-found |
| CrewAI mock LLM | Prompt/messages and declared tool schemas | Deterministic text/tool-call response |
| Crew response schema | Draft response fields | Validated structured response |
| Input guardrails | Raw request text | Masked safe text or rejection |
| Output groundedness guard | Draft and retrieved context | Pass or refusal/fallback |
| Session memory | Session ID and messages | Session-isolated history |
| AutoGen reviewer | Draft and original context | Structured verdict |
| Budget guard | Request text and cap | Estimate and allow/reject result |
| Grounded-generation cache | Normalized query (trimmed, punctuation-stripped at the ends, lower-cased) | Cached result or miss |
| Request logger | Safe request metadata and timings | One JSONL record |
| FastAPI routes | Validated HTTP/WebSocket payload | Structured response |

---

## 6. Structured data models

Define Pydantic models for at least the following boundaries.

### 6.1 API request

Suggested fields:
- `query: str`
- `session_id: str`
- optional `record_id: str`
- optional request metadata that does not contain secrets.

The implementation may use separate request models for policy questions and record lookups. Validate that a request is not ambiguous or malformed.

### 6.2 Crew response

Define one explicit response model used for every final CrewAI crew response. Include enough fields to represent:
- answer text,
- response type/intent,
- evidence or source document identifiers,
- fallback/unsupported status,
- optional application lookup result.

Validate every response in code.

### 6.3 AutoGen verdict

Required fields:
- `approved: bool`
- `final_answer: str`
- `reason: str`

Use this model as `output_content_type` for the Final-Editor agent and register its structured message type with the team.

### 6.4 API response

Return a stable schema containing:
- trace ID,
- final answer,
- structured status/result metadata,
- sources/evidence where applicable,
- fallback or rejection status where applicable.

Do not expose internal secrets, raw phone PII, or unnecessary internal prompts.

---

## 7. CrewAI `MOCK_LLM` design constraints

This is a high-risk integration point and must be prototyped before the full crew is built.

1. Implement a concrete subclass of `crewai.llms.base_llm.BaseLLM`.
2. Provide deterministic responses based on the current task/prompt and tool schemas.
3. Use CrewAI's declared tool argument schema to determine the expected argument shape.
4. Never infer argument shape from a substring in the tool name.
5. Parse tool observations only from the model-generated output/turn, not from the entire system prompt or template.
6. Include explicit tests for:
   - a RAG tool whose name contains `lookup`, such as `rag_lookup`;
   - the application lookup tool;
   - first ReAct turn before any tool observation exists;
   - a tool result followed by a final answer.
7. Verify that the right tool is called with the right structured arguments.
8. Disable telemetry before `.kickoff()`.

The mock implementation must not require a live provider or network request.

---

## 8. AutoGen mock-review design constraints

The AutoGen review stage must also work without an external LLM service.

- Provide a deterministic mock model/client integration compatible with the installed AutoGen version, or another supported local deterministic mechanism that still executes the required AutoGen two-agent team.
- Do not replace the required AutoGen `RoundRobinGroupChat` stage with a hand-written if/else function that bypasses the team.
- Use two agents with the required reviewer/editor responsibilities.
- Use `max_turns=2`.
- Configure the Final-Editor `output_content_type` with the verdict model.
- Register `StructuredMessage[VerdictModel]` using `custom_message_types`.
- Pass the actual CrewAI draft and original retrieved context.
- Demonstrate one unchanged approval and one revision of an intentionally ungrounded draft.
- Capture the structured verdict and final answer for both tests.

Before implementing the complete stage, create a small compatibility spike that instantiates the team and produces one structured verdict under offline mock execution. Pin the compatible dependency versions only after this spike passes.

---

## 9. Caching, budgets and ordering constraints

### 9.1 Request processing order

Recommended order:
1. Request validation.
2. Trace ID/session setup.
3. Input PII masking.
4. Prompt-injection detection.
5. Runtime budget check.
6. Session-history retrieval.
7. CrewAI orchestration.
8. Grounded-generation cache/retrieval inside the RAG service.
9. Pydantic crew-response validation.
10. Output groundedness check.
11. AutoGen review.
12. Final response validation.
13. Session-history update.
14. PII-safe structured logging.
15. Return response.

The exact placement of safe logging can be implemented with middleware/finally handling, but every request must generate one JSONL entry and raw phone PII must never be logged.

### 9.2 Cache scope

Cache only the grounded-generation result, keyed by normalized query text. The normalization rule is: strip leading and trailing whitespace and punctuation, convert to lower case, and collapse internal whitespace runs to one space; internal punctuation (such as the hyphen in a record ID) is preserved. The rule is lexical only and no semantic-equivalence claim is made. A query that is empty after normalization is rejected by input validation before the cache. A cache hit must avoid the redundant grounded-generation call. It must not bypass:
- PII masking,
- prompt-injection detection,
- budget enforcement,
- session-specific memory,
- response schema validation,
- output groundedness validation,
- AutoGen review.

Track hit/miss counters and grounded-generation invocation count.

### 9.3 Budget scope

The runtime budget must be checked per request before expensive orchestration. In `MOCK_LLM`, use a deterministic token-estimation rule and a configured cap. Document that this is a simulated operational budget, not a real provider charge. Demonstrate an oversized request rejected before crew execution.

---

## 10. Observability and evidence

Every request must create one JSONL record with:
- `trace_id`
- timestamp
- duration/latency
- endpoint/request type
- safe, masked request text or a safe request summary
- response status
- error/rejection code where applicable
- cache hit/miss information where useful

Never log raw fixed-format phone PII. Use the same masking function for model input and log input.

Capture separate text transcripts/reports for:
- category/status/priority distribution;
- similarity calibration values;
- five in-scope RAG queries and one out-of-scope fallback;
- per-query precision/recall for both chunking strategies;
- RAG and lookup tool invocation;
- multi-turn memory and fresh session;
- each guardrail firing;
- WebSocket disconnect handling;
- all 15 evaluation queries and four averages;
- AutoGen unchanged approval and revision;
- least-autonomy enforcement;
- oversized budget rejection;
- cache miss followed by cache hit.

---

## 11. Technology and dependency decisions

Required technologies:
- Python.
- CrewAI.
- AutoGen AgentChat/Core APIs supporting `RoundRobinGroupChat`.
- FastAPI.
- Pydantic.
- LangChain session-based memory APIs as specified.
- SentenceTransformers local embeddings.
- ChromaDB local vector store.

Implementation notes:
- Verify compatible package versions for CrewAI, AutoGen, LangChain, Pydantic, FastAPI, SentenceTransformers and ChromaDB in a clean environment.
- Do not blindly install latest versions if APIs conflict.
- Record tested versions in the dependency manifest and README.
- Avoid deprecated or undocumented integration shortcuts where the capstone brief specifies a concrete API.
- Keep embedding-model assets locally available before offline graded runs.
- Do not require Docker, cloud hosting, hosted databases or paid accounts unless purely optional.

---

## 12. Architecture acceptance checklist

- [ ] Every logical component has a clear responsibility and contract.
- [ ] Data and knowledge layer is separate from orchestration and API.
- [ ] Both chunking strategies are independently indexed.
- [ ] Only the selected RAG collection is wired into the production crew.
- [ ] Retrieval Agent has only the RAG tool.
- [ ] Lookup Agent alone has the application lookup tool.
- [ ] Composer has no direct lookup capability.
- [ ] CrewAI uses a `BaseLLM` subclass for `MOCK_LLM`.
- [ ] AutoGen team executes in mock mode with registered structured messages.
- [ ] Guardrails execute before model input and before final output delivery.
- [ ] Cache is scoped to grounded generation and does not bypass controls.
- [ ] Cache-key normalization follows the documented lexical rule.
- [ ] Budget is enforced before crew execution.
- [ ] API HTTP and WebSocket paths share the same safety and orchestration services.
- [ ] Phone PII is masked before model input and disk logging.
- [ ] WebSocket disconnect is handled without server failure.
- [ ] WebSocket concurrency uses async patterns, and session memory is isolated per connection/session ID.
- [ ] Each request creates one safe JSONL log event.
- [ ] All architecture decisions are compatible with zero-key, zero-network graded execution.

---

## 13. Implementation sequence

This document is used together with:
- `naukri_capstone_requirements.md`
- `naukri_capstone_implementation_plan.md`

The requirements document is authoritative; this architecture document defines the component boundaries and flows. Implementation does not build the whole repository in one pass: it first inspects the requirements, produces a traceability matrix and runs the staged compatibility spikes, then proceeds through the implementation plan and runs the acceptance tests after each stage.
