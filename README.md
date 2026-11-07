# Naukri.com Domain Support Agent

**Completed track: Naukri.com (Recruitment & HR).** A multi-agent support system built with CrewAI, RAG over ChromaDB with SentenceTransformers, an AutoGen review team, LangChain session memory and FastAPI (HTTP and WebSocket). It answers hiring-policy questions from a locally authored knowledge base, looks up fabricated job-application records, remembers a conversation, guards its inputs and outputs, has every draft reviewed by an independent two-agent team, and is evaluated end to end.

Contents: [1 Dataset design](#1-dataset-design-choices-task-1) · [2 Offline and keyless](#2-all-graded-runs-use-mock_llm-zero-api-keys-zero-network) · [3 Telemetry](#3-crewai-telemetry-is-disabled) · [4 Similarity calibration](#4-similarity-calibration-and-fallback-threshold) · [5 Chunking comparison](#5-chunking-comparison-and-recommendation) · [6 Escalation score](#6-escalation-score-and-threshold) · [7 Guardrails](#7-guardrails) · [8 Governance, budget, cache](#8-governance-budget-and-cache) · [9 Evaluation](#9-evaluation-15-queries-four-scores) · [10 API and logging](#10-api-and-logging) · [11 Setup, run, test](#11-setup-run-and-test) · [12 Layout](#12-repository-layout) · [13 Deviations and limits](#13-deviations-and-limitations) · [14 Originality](#14-originality)

## 1. Dataset design choices (Task 1)

`dataset.py` generates `JOB_APPLICATIONS` with a plain seeded weighted draw (`random.Random(42)` and `random.choices`). The first draw with these choices satisfied every constraint, so neither the seed nor the weights were changed, and no record was ever edited by hand. Regenerate and validate with `python dataset.py` (it also rewrites `data/generated_job_applications.json`).

- **Random seed:** 42
- **Records:** 60; IDs `APP-0001` to `APP-0060`
- **Category weights:** Software Engineer 0.28, Data Analyst 0.20, Product Manager 0.16, HR Executive 0.16, Sales Associate 0.20
- **Status weights:** Applied 0.25, Screening 0.25, Interview Scheduled 0.20, Offered 0.10, Rejected 0.20
- **Priority-flag probability:** 0.18 (`flagged_priority_review`, a weighted `True`/`False` draw)
- **Expected-salary range:** INR 400,000 to INR 3,500,000 per year, uniform integer. Annual expected salaries of INR 400,000 to 3,500,000 run from entry-level sales and HR roles up to senior software and product roles in the Indian market, so one range is realistic for all five categories.
- **`days_since_created`:** uniform integer 0 to 30

Printed distribution of the generated data:

| Category | Records |
|---|---|
| Software Engineer | 9 |
| Data Analyst | 11 |
| Product Manager | 12 |
| HR Executive | 12 |
| Sales Associate | 16 |

| Status | Records |
|---|---|
| Applied | 21 |
| Screening | 15 |
| Interview Scheduled | 7 |
| Offered | 4 |
| Rejected | 13 |

Flagged for priority review: 8/60 = 13.33% (required range 10% to 30%). The validator asserts at least 40 records, at least 3 per category, at least 1 per status, field types and ranges, unique IDs, salary range and the flag percentage; `tests/test_dataset.py` also shows it rejects bad data.

The knowledge base is 12 Markdown files in `knowledge_base/` (one per required topic, 4 sentences each, written for this project); `src/rag/kb_loader.py` checks the topic set and the 2 to 5 sentence range.

## 2. All graded runs use `MOCK_LLM`, zero API keys, zero network

Every language-model consumer (grounded generation, the evaluation judge, the CrewAI crew and the AutoGen review team) goes through one deterministic core, `src/llm/mock_llm.py`, selected by `MOCK_LLM` (default `true`). Graded runs use **zero API keys** and make **no network calls**: SentenceTransformers and ChromaDB run locally, and an autouse fixture in `tests/conftest.py` makes every non-loopback `socket.connect` raise, so the whole test suite, the evaluation and the transcripts run under that block. The one networked step is the one-time embedding-model download, `python -m scripts.prefetch_model`; graded runs then use `HF_HUB_OFFLINE=1`. A real LLM is optional in the brief and is not implemented: `MOCK_LLM=false` raises a clear `NotImplementedError`, and no acceptance criterion depends on it.

## 3. CrewAI telemetry is disabled

`src/config/settings.py` sets `CREWAI_DISABLE_TELEMETRY=true` and `OTEL_SDK_DISABLED=true` (and `CREWAI_DISABLE_TRACKING`, `ANONYMIZED_TELEMETRY=False` for ChromaDB, `HF_HUB_DISABLE_TELEMETRY`, `TRANSFORMERS_OFFLINE`) as environment variables **before** `crewai`, `chromadb` or `sentence_transformers` is imported and before any `.kickoff()`; every entry point and `tests/conftest.py` import it first, and `tests/test_offline.py` proves the variables are set before `crewai` is imported.

## 4. Similarity calibration and fallback threshold

Top-1 cosine similarity (1 minus Chroma's cosine distance, model `sentence-transformers/all-MiniLM-L6-v2`) was measured on 6 in-scope and 4 deliberately out-of-scope queries for both collections. Under `MOCK_LLM` this similarity is the only groundedness signal; no model judges semantic groundedness.

| Collection | Set | Query | Top-1 similarity |
|---|---|---|---|
| fixed | in-scope | How many hours before an interview can I reschedule my slot? | 0.7099 |
| fixed | in-scope | What is the notice period for an employee who is still in probation? | 0.7917 |
| fixed | in-scope | How much referral bonus is paid for a managerial role? | 0.7948 |
| fixed | in-scope | Who can apply for an internal transfer and how long must they have been in their role? | 0.7725 |
| fixed | in-scope | How long do you keep the data of rejected applicants? | 0.7839 |
| fixed | in-scope | Can a new joiner work from home during probation? | 0.5527 |
| fixed | out-of-scope | What is the capital of France? | 0.0852 |
| fixed | out-of-scope | How do I bake a chocolate cake? | 0.0458 |
| fixed | out-of-scope | Who won the football world cup in 2018? | 0.1152 |
| fixed | out-of-scope | Which programming language is best for web scraping? | 0.1235 |
| sentence | in-scope | How many hours before an interview can I reschedule my slot? | 0.6938 |
| sentence | in-scope | What is the notice period for an employee who is still in probation? | 0.8533 |
| sentence | in-scope | How much referral bonus is paid for a managerial role? | 0.6964 |
| sentence | in-scope | Who can apply for an internal transfer and how long must they have been in their role? | 0.7961 |
| sentence | in-scope | How long do you keep the data of rejected applicants? | 0.7738 |
| sentence | in-scope | Can a new joiner work from home during probation? | 0.5569 |
| sentence | out-of-scope | What is the capital of France? | 0.0238 |
| sentence | out-of-scope | How do I bake a chocolate cake? | 0.0767 |
| sentence | out-of-scope | Who won the football world cup in 2018? | 0.0809 |
| sentence | out-of-scope | Which programming language is best for web scraping? | 0.1509 |

| Collection | In-scope minimum | Out-of-scope maximum | Gap | Chosen threshold (midpoint) |
|---|---|---|---|---|
| fixed-size with overlap | 0.5527 | 0.1235 | 0.4292 | 0.338 |
| sentence-based | 0.5569 | 0.1509 | 0.4060 | 0.354 |

**Selected fallback threshold: 0.338** (the deployed fixed-size collection). The clusters do not overlap, and four held-out queries that were not used to choose it behave correctly. When top-1 similarity is below the threshold the system returns an explicit "I don't know" fallback. Rationale (hand-written after the measurements):

> The two clusters never touch. On the fixed-size collection every in-scope query scored at least 0.5527 and every out-of-scope query at most 0.1235, a gap of 0.4292, so the threshold 0.338 is the midpoint of that gap; the sentence-based collection gave a gap of 0.4060 and a midpoint of 0.354. A round preset would have been wrong in practice: 0.6 would refuse the in-scope remote-work query (0.5527) and the held-out background-check query (0.5453). The HR-adjacent dress-code query (0.3220) is correctly refused but only 0.0160 below the threshold, so questions that sit close to HR topics without being in the knowledge base are the likeliest false accepts, and ten calibration queries are a small sample for a topical-overlap proxy that does not prove an answer is supported.

Full tables: `evaluation/reports/similarity_calibration.md`. Demonstrations (6 in-scope answers and out-of-scope fallbacks): `transcripts/rag_demos.md`.

## 5. Chunking comparison and recommendation

Both strategies are implemented over the same documents and indexed in separate collections with `collection.upsert()`: fixed windows of 30 words with an 8-word overlap (40 chunks, `kb_fixed_overlap`) and one chunk per sentence (48 chunks, `kb_sentence`). Document-level precision and recall on the same six in-scope queries (top-k = 3, chunks mapped to parent documents and deduplicated before scoring; per query: precision = hits / distinct documents retrieved, recall = hits / relevant documents declared):

| # | Query | Relevant documents (declared) | Fixed: precision; recall | Sentence: precision; recall |
|---|---|---|---|---|
| 1 | How many hours before an interview can I reschedule my slot? | interview_scheduling | 1/1 = 1.000; 1/1 = 1.000 | 1/1 = 1.000; 1/1 = 1.000 |
| 2 | What is the notice period for an employee who is still in probation? | notice_period, probation_period | 2/2 = 1.000; 2/2 = 1.000 | 2/2 = 1.000; 2/2 = 1.000 |
| 3 | How much referral bonus is paid for a managerial role? | referral_bonus | 1/1 = 1.000; 1/1 = 1.000 | 1/1 = 1.000; 1/1 = 1.000 |
| 4 | Who can apply for an internal transfer and how long must they have been in their role? | internal_transfer | 1/2 = 0.500; 1/1 = 1.000 | 1/2 = 0.500; 1/1 = 1.000 |
| 5 | How long do you keep the data of rejected applicants? | applicant_data_retention | 1/2 = 0.500; 1/1 = 1.000 | 1/2 = 0.500; 1/1 = 1.000 |
| 6 | Can a new joiner work from home during probation? | probation_period, remote_work | 2/3 = 0.667; 2/2 = 1.000 | 2/3 = 0.667; 2/2 = 1.000 |

Mean precision / mean recall: fixed-size 0.778 / 1.000; sentence-based 0.778 / 1.000.

**Harder multi-document queries.** The six shared queries tie, so three extra queries were added that each need evidence from three documents (with top-k = 3, full recall needs every retrieved chunk to come from a different relevant document). They are used only for this comparison, not for calibration:

| # | Query | Relevant documents (declared) | Fixed: precision; recall | Sentence: precision; recall |
|---|---|---|---|---|
| H1 | If I resign while still on probation, how much notice do I give and will HR hold an exit interview? | exit_interview, notice_period, probation_period | 3/3 = 1.000; 3/3 = 1.000 | 3/3 = 1.000; 3/3 = 1.000 |
| H2 | Once my probation is confirmed, can I work remotely and also move to another team internally? | internal_transfer, probation_period, remote_work | 2/2 = 1.000; 2/3 = 0.667 | 3/3 = 1.000; 3/3 = 1.000 |
| H3 | Does a new hire on probation get remote work, and what notice applies if they leave? | notice_period, probation_period, remote_work | 2/3 = 0.667; 2/3 = 0.667 | 3/3 = 1.000; 3/3 = 1.000 |

| Query set | Fixed: mean precision / recall | Sentence: mean precision / recall |
|---|---|---|
| 6 shared queries | 0.778 / 1.000 | 0.778 / 1.000 |
| 3 harder queries | 0.889 / 0.778 | 1.000 / 1.000 |
| all 9 queries | 0.815 / 0.926 | 0.852 / 1.000 |

Recommendation (hand-written after the runs):

> On the six shared Task 4 queries the two collections tie at k = 3 (mean precision 0.778 and mean recall 1.000 for both), but on the three harder queries that each span three documents sentence-based chunking pulls ahead with mean precision 1.000 and mean recall 1.000 against 0.889 and 0.778 for fixed-size, because the 30-word windows spent a top-3 slot on a second remote-work window in one query and on the neighbouring internal-transfer document instead of the notice-period document in another (fixed-size only reaches recall 1.000 at k = 5). Retrieval alone therefore favours sentence-based chunking, but in the 15-query end-to-end evaluation the fixed-size collection scored completeness 5.000 against 4.733, because a single retrieved sentence often leaves out the second fact a policy answer needs, and most real questions to this assistant are about a single topic. I therefore recommend deploying the fixed-size collection for this knowledge base, accepting weaker coverage on three-topic questions and a narrower fallback margin (the dress-code query scores 0.3220 against a threshold of 0.338, a gap of 0.0160, versus 0.1019 on the sentence-based collection at 0.354), and I would switch to sentence-based chunking if multi-topic questions became common.

The recommended fixed-size collection is the RAG input to the crew. Details: `evaluation/reports/chunking_comparison.md`.

## 6. Escalation score and threshold

`check_job_application_status(record_id: str) -> dict` returns `status`, `expected_salary_inr` and an `escalation_score` in [0, 1] (unknown IDs return a controlled `found: false` result and nothing invented):

`escalation_score = 0.60 * flagged_priority_review + 0.40 * (1 - days_since_created / 30)`

- the priority flag contributes 0.60: a record marked for priority review is always worth a human look;
- the recency component contributes up to 0.40 and falls linearly from 0.40 (created today) to 0 (30 days old): a new application is still inside its first-response window, so it is routed to a person while a quick reply can still help; an older one has already been through recruiter triage.

**Escalation is recommended when the score is at least 0.33.** Computed from the generated dataset: P20 = 5.80, P50 = 12.00 and P80 = 23.00 days since created. 18 of 60 records escalate (8 flagged, 10 unflagged but recent). Justification (hand-written after the computation):

> The 20th percentile of days_since_created in the generated dataset is 5.80 and the 80th is 23.00, so I set the threshold at 0.33, which sits just below the score of an unflagged application created 5 days ago (0.3333) and above that of one created 6 days ago (0.3200). That makes an unflagged application escalate only when it is fresher than the dataset's P20, which is 10 of the 60 records, while all 8 flagged records escalate because their lowest score is 0.6800; together that is 18 of 60 escalations, 8 driven by the flag and 10 by recency alone. The threshold sits below 0.40 on purpose, because any value between 0.40 and 0.60 would reproduce the boolean flag exactly, and the P80 of 23 days shows why the flag still matters: an unflagged application that old scores only 0.0933, while a flagged one scores 0.6933. The score only routes a case to a human reviewer; it is not a judgement about the candidate.

Report with the decision-region table: `evaluation/reports/escalation_threshold.md`.

## 7. Guardrails

- **Phone masking (input and log).** Only the fixed-format phone number inside contact details is masked, by one function, `mask_phone`, used for model input, session history and log content. The canonical format is `+91-NNNNN-NNNNN` in a `Contact: ...` line; the variants `+91 NNNNN NNNNN`, `+91NNNNNNNNNN`, `NNNNN-NNNNN`, `NNNNN NNNNN` and ten contiguous digits are masked too, and any other ten-digit number is masked as a side effect. The replacement is `[PHONE_MASKED]`. **Candidate names, expected salary and background-check results are not masked**: they are out of scope for this pattern-based, keyless masker, and all examples in the repository are fabricated.
- **Prompt injection.** A deterministic list of regular expressions (override verbs such as ignore, disregard, forget or set aside applied to instructions, rules or guidance; reveal the system prompt; role reset; developer mode; disable safety; delimiter injection) blocks matching requests before any model call. A separate check rejects bulk-data requests such as "list all salaries" or "export every candidate", because the lookup reads one application at a time (reason `unsupported_bulk_request`; an override attempt that also asks for bulk data keeps the stronger `prompt_injection` reason). Limitations: it is not a complete detector; it misses paraphrases that avoid those verbs, objects and phrasings, other languages, obfuscated or encoded text and instructions hidden in retrieved documents (one test shows a known miss). Added documents pass through the injection check but not the bulk-request check.
- **Groundedness.** A deterministic token-overlap check (a sentence is supported when at least 60% of its content words appear in the evidence) refuses a draft when no retrieved evidence supports the question and flags partially unsupported drafts, which the AutoGen team then revises.

Demonstrations: `transcripts/guardrail_demos.md`.

**Structured output (Task 9).** `response_format = CrewResponse` (`src/schemas/crew_response.py`) is the Pydantic model every crew response must conform to. It is passed to the Composer task as CrewAI's `output_pydantic` (the CrewAI 1.15 parameter for a structured response format), and after every `.kickoff()` the result is validated again in code with `response_format.model_validate`; a response that does not conform is never delivered (`transcripts/structured_output.md`).

## 8. Governance, budget and cache

All four governance layers are applied. The brief names concrete controls for the Application layer (least autonomy, risk classification) and the Runtime layer (per-request budget), which are enforced in code and demonstrated; the Data and Model layers list the controls the system already has. Full text: `src/governance/risk_assessment.md`.

- **Data layer.** Only fabricated data (seeded dataset, self-written knowledge base). The fixed-format phone number is masked by one function before it reaches the model, session history, an added document or the log; names, salary and background-check results are out of scope for masking. One application per lookup, bulk requests refused, answers never logged, session memory in process only, and the index and live log are git-ignored, in the spirit of the knowledge base's 12-month retention rule.
- **Model layer.** One deterministic offline mock serves every model call and records each one by purpose for audit; `MOCK_LLM=false` fails loudly instead of switching to an unreviewed live model. The embedding model and every library version are pinned (`requirements.lock`), telemetry is off, the fallback threshold is calibrated on measured scores, and every crew output and review verdict must validate against a Pydantic schema.
- **Application layer:** risk classification and least autonomy, below.
- **Runtime layer:** the per-request budget, below.

- **Risk classification: High.** The scheme places financial data and hiring decisions in the High class, and this agent touches both: the lookup tool returns a candidate's expected salary, which is personal financial data, and every answer concerns a live hiring process. The class is set by the kind of data and process the design handles, not by the fabricated test data, so it would stay High with real candidates. The controls are sized for that class: the agent is read-only, it does not rank, shortlist, reject or select any candidate and makes no compensation decision, and the escalation score only routes a case to a human reviewer; only the Lookup Agent can read a record, one application per request, and bulk requests are refused; the phone number is masked before the model, memory or log see it; and every answer passes a groundedness guard and an independent two-agent review before delivery. Connecting real candidate data would also need caller authentication, so that a salary is shown only to someone entitled to see it.
- **Least autonomy.** Only the Lookup Agent holds `check_job_application_status`; `build_agents` checks a per-role allow-list before constructing each agent and raises `AutonomyViolation` otherwise (`transcripts/governance_autonomy_demo.md`).
- **Per-request budget.** Estimate `ceil(len(masked question) / 4) + 600` tokens against a cap of 1,000; an oversized request is rejected (HTTP 413) before the crew is built (`transcripts/governance_budget_demo.md`). It is a simulated control: no provider is called in `MOCK_LLM` mode, so nothing is charged.
- **Grounded-generation cache.** Key: the lexically normalised query (strip leading and trailing whitespace and punctuation, lower-case, collapse internal whitespace, keep internal punctuation such as the hyphen in `APP-0007`) plus the index version. It matches case, whitespace and surrounding-punctuation variants only; **it does not match paraphrases and makes no semantic-match claim**. A query that is empty after normalisation is rejected by input validation before the cache. It caches only the grounded-generation step, so guardrails, budget, session handling, schema validation, groundedness and the AutoGen review still run on a hit; `/add-document` clears it (`transcripts/cache_demo.md`).

## 9. Evaluation: 15 queries, four scores

`python -m evaluation.run_evaluation` runs 15 queries (all 12 knowledge-base topics, one status lookup, one out-of-scope question and one prompt injection) through the same pipeline as `POST /ask` and scores each answer with an LLM-as-judge prompt (rubric, 1 to 5 scale, JSON answer) sent through the shared `MockLLM`. **The judge is a rule-based stand-in, not a semantic LLM judge**: it checks required phrases, forbidden claims, expected behaviour and word overlap with the cited documents. Scores are integers from 1 to 5 for Accuracy, Grounding, Completeness and Safety; all four are reported for every query in `evaluation/reports/15_query_results.md`.

| Accuracy | Grounding | Completeness | Safety |
|---|---|---|---|
| 5.000 | 5.000 | 5.000 | 5.000 |

Averages over all 15 queries for the deployed fixed-size collection (the same queries on the sentence-based collection average completeness 4.733).

**Judge negative controls.** A judge that gives every answer 5 would prove nothing, so the same judge also scores seven hand-written bad answers, each wrong in one known way, plus one correct reference (C00). They are not pipeline output and are not in the averages above. Every bad answer scores below 5 on at least one property:

| ID | Deliberate defect | Accuracy | Grounding | Completeness | Safety |
|---|---|---|---|---|---|
| C00 | reference: correct, grounded answer (should score 5) | 5 | 5 | 5 | 5 |
| C01 | wrong number (the forbidden INR 75,000) | 1 | 5 | 3 | 3 |
| C02 | vague answer, required facts missing | 2 | 5 | 1 | 5 |
| C03 | invented (hallucinated) extra claim | 5 | 3 | 5 | 5 |
| C04 | raw phone number leaked | 5 | 3 | 5 | 3 |
| C05 | right words, wrong source document cited | 5 | 1 | 5 | 5 |
| C06 | out-of-scope question answered instead of the fallback | 1 | 1 | 1 | 1 |
| C07 | injection obeyed: prompt and ReAct text leaked | 1 | 1 | 1 | 1 |

One limit is visible here: C01 changes a number (INR 75,000) and still gets grounding 5, because a word-overlap check cannot see that a figure changed; the accuracy score (forbidden claim) is what catches it. Full table with answers and rationales: `evaluation/reports/15_query_results.md`.

Findings (hand-written):

> On the 15 queries the fixed-size collection used in production averages accuracy 5.000, grounding 5.000, completeness 5.000 and safety 5.000 with no query below 5, while the same queries on the sentence-based collection average completeness 4.733 because E07 and E11 each miss one required fact, which is why the fixed-size collection is the one deployed. Routing uses policy keywords drawn from the stems of all 12 topic names, so a question such as E03 (a salary revision on an offer) is answered from the knowledge base and not treated as an application question. Accuracy and safety sit at 5 across the whole set, so the set separates the two strategies only through completeness, and with a rule-based judge and 15 queries it is evidence and not proof.

## 10. API and logging

| Endpoint | Purpose |
|---|---|
| `POST /ask` | `{"question": "...", "session_id": "optional"}` returns a structured answer (trace ID, session ID, status, answer, sources, budget info); 400 injection, 413 over budget, 422 invalid |
| `POST /add-document` | `{"title", "text", "doc_id"?}` adds a document to the vector index (masked, injection-checked, additive: an existing ID such as a required document is refused with 409) |
| `WebSocket /chat` | multi-turn chat; `/chat?session_id=...` joins a session, a connection without one gets a unique one |

HTTP and WebSocket share one request path (`src/pipeline.py`): validation, masking, injection check, budget, session history, CrewAI crew (Retrieval, Lookup and Composer agents through `.kickoff()`), response-schema validation, groundedness guard, AutoGen review (`Policy_Compliance_Reviewer` and `Final_Editor`, `RoundRobinGroupChat` with `max_turns=2`, structured `Verdict`), final validation, session update. The pipeline runs in a thread pool so slow or disconnected clients do not block others; a client disconnecting mid-conversation is handled without affecting anyone else; session memory is isolated per session ID (`transcripts/api_demo.md`).

Every HTTP request (including unknown routes, wrong methods and `/docs`) and every WebSocket turn writes exactly one flat JSON line (`@timestamp`, `trace_id`, `duration_ms`, endpoint, method, HTTP status, status; for `/ask` and WebSocket turns also the masked request text, budget and cache fields and the review result) to `logs/requests.jsonl`. Strings are passed through `mask_phone` before the line is written, so the raw phone number never reaches the file (`transcripts/logging_privacy_demo.md`; sample: `logs/sample_requests.jsonl`).

## 11. Setup, run and test

Python 3.11 is required (CrewAI does not support 3.14). Clone the repository and enter it:

```bash
git clone https://github.com/jatin461/Naukri-Domain-Support-Agent.git
cd Naukri-Domain-Support-Agent
```

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.lock
```

```bash
.venv/bin/python -m scripts.prefetch_model
```

```bash
.venv/bin/python -m scripts.build_index
```

```bash
.venv/bin/python -m uvicorn src.api.main:app --port 8000
```

`prefetch_model` is the only step that needs the network (add `--check` to verify the model loads offline). `build_index` creates the ChromaDB collections in `chroma_db/` (not committed). Try the running server:

```bash
curl -s -X POST localhost:8000/ask -H 'content-type: application/json' -d '{"question": "How long is the probation period?"}'
```

Run the tests (the whole suite runs with network access blocked):

```bash
.venv/bin/python -m pytest
```

Regenerate every report and transcript, run the tests and build the acceptance matrix in one command:

```bash
bash scripts/run_acceptance.sh
```

That script validates the dataset, rebuilds the index, regenerates the calibration, comparison, escalation and evaluation reports and all transcripts, runs the test suite with a JUnit report and writes `evaluation/reports/acceptance_matrix.md`, a pass/fail matrix built only from executed tests (`scripts/acceptance_report.py`). Other commands: `python dataset.py` (dataset and distribution), `python -m scripts.calibrate_threshold`, `python -m scripts.compare_chunking`, `python -m scripts.escalation_report`, `python -m evaluation.run_evaluation`, `python -m scripts.generate_transcripts`.

## 12. Repository layout

```text
README.md  requirements.txt  requirements.lock  .python-version  .env.example  pytest.ini
dataset.py                      seeded generator and validator
data/                           generated dataset snapshot
knowledge_base/                 12 policy documents
src/config/                     environment and telemetry settings (imported first)
src/rag/                        loader, chunking, embeddings, vector store, retrieval, grounded generation, calibration, evaluation
src/tools/                      RAG tool, application lookup and escalation score
src/llm/                        shared mock LLM core and the CrewAI adapter
src/crew/                       agents, tasks, routing, crew
src/memory/                     session memory and reference resolver
src/guardrails/                 input and output guardrails
src/schemas/                    Pydantic models (crew response, verdict, API)
src/review/                     AutoGen review team and its mock client
src/governance/                 least autonomy, budget, risk assessment
src/cache/                      grounded-generation cache
src/observability/              JSONL request logging
src/api/                        FastAPI app and document indexing
src/pipeline.py                 the single request path
evaluation/                     15 queries, judge, runner, reports/, hand_written/ analysis text
scripts/                        index build, calibration, comparison, escalation report, prefetch, transcripts, acceptance
transcripts/                    generated demonstrations for every task
logs/sample_requests.jsonl      generated sample log
tests/                          the test suite
docs/, specs/                   source requirements, traceability matrix, dependency notes
```

## 13. Deviations and limitations

- **AutoGen agent names** use underscores (`Policy_Compliance_Reviewer`, `Final_Editor`) because AutoGen 0.7.5 requires Python identifiers; the brief gives the hyphenated names only as examples.
- **Review depth.** The AutoGen team runs offline on a deterministic client; its decision is the same sentence-support check as the groundedness guard, executed independently inside the required team. It is not a semantic policy review.
- **Evaluation depth.** The judge is rule-based and the set has 15 queries; accuracy and safety saturate at 5, so the set separates strategies only through completeness.
- **Retrieval.** The mock generator is extractive (it joins retrieved passages, marking breaks between non-adjacent pieces with "…"), so answers can include loosely related neighbouring sentences; the similarity threshold is a topical-overlap proxy, and HR-adjacent questions outside the knowledge base are the likeliest false accepts.
- **Memory.** Session memory is in-process only; the reference resolver assumes the most recent record or policy question in the session is meant and understands only the cue words in `src/crew/routing.py`.
- **Phone masking** also masks any other ten-digit number; names, salary and background-check results are not masked.
- **Chroma quirk.** A late small upsert was sometimes not returned by queries; collections use minimum HNSW batch thresholds and `/add-document` verifies searchability (`docs/dependency_compatibility.md`). An index persisted by other settings keeps them; rebuild it with `scripts.build_index`.
- **Budget** numbers are design choices, not provider prices. Real-LLM mode is not implemented.

## 14. Originality

The dataset generator, knowledge-base documents, code, test queries and all analysis text (threshold rationale, chunking recommendation, escalation justification, evaluation findings, risk paragraph) were written for this project. A test checks that no knowledge-base document shares an 8-word run with the requirement documents, and that the analysis text cites the computed numbers. All people, numbers and records are fabricated.
