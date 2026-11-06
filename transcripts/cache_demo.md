# Grounded-generation cache

Key: `(index version, normalised query)`; normalisation strips leading and trailing whitespace and punctuation, lower-cases and collapses internal whitespace, and keeps internal punctuation (`APP-0007` keeps its hyphen). It is purely lexical: it matches variants that differ only in case, surrounding whitespace or surrounding punctuation, and it does not match paraphrases.

## Normalisation examples

| Input | Normalised key |
|---|---|
| `  What is the NOTICE period? ` | `what is the notice period` |
| `what is the notice period?!` | `what is the notice period` |
| `How long is the notice period` | `how long is the notice period` |
| `status of APP-0007?` | `status of app-0007` |
| `???` | `` |

## Before / after (same process, same pipeline)

| Request | cache event | grounded-generation LLM calls | retrievals | cache hits / misses |
|---|---|---|---|---|
| first: `How long is the probation period?` | ['miss'] | 1 | 1 | 0 / 1 |
| repeat, different case and trailing punctuation: `  how long is the PROBATION period?! ` | ['hit'] | 1 | 1 | 1 / 1 |
| paraphrase (misses: no semantic matching): `What is the length of probation for new hires?` | ['miss'] | 2 | 2 | 1 / 2 |

- first and repeated answers identical: True; both validated as `CrewResponse`: True
- the repeated request still ran every stage: `['validate', 'trace_session', 'mask_pii', 'injection_check', 'budget_check', 'session_history', 'resolve_reference', 'crew', 'validate_response', 'groundedness', 'autogen_review', 'validate_final', 'session_update']`; AutoGen review verdict approved: True; budget estimate: 609 tokens

## An empty query after normalisation is rejected before the cache

- `???` -> status rejected, reason invalid_request; stages ['validate']; cache stats unchanged: True

## Adding a document invalidates the cache

- index version 0 -> 1; cache size after the add: 0
