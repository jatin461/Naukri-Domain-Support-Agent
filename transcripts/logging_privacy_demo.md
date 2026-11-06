# Logging and privacy demonstration

Log file: `logs/sample_requests.jsonl` (26 entries: every request of the API demo plus three requests that contain the fabricated phone number +91-98765-43210).

## One flat JSON object per request (ELK style)

```json
{"@timestamp": "2026-11-06T21:18:07.153Z", "level": "INFO", "event": "request", "trace_id": "0a6ab53e5c3749f1b5d2acca03d7aad0", "session_id": "sess-0b083108415c", "endpoint": "/ask", "method": "POST", "status": "answered", "http_status": 200, "duration_ms": 68.784, "request_text": "How much referral bonus is paid for a managerial role?", "phone_numbers_masked": 0, "rejection_reason": null, "injection_matched": "", "resolution": "unchanged", "groundedness_action": "pass", "estimated_tokens": 614, "token_limit": 1000, "budget_allowed": true, "simulated_cost_usd": 0.001228, "cache_hits": 0, "cache_misses": 1, "review_approved": true, "review_revised": false, "tools": "rag_lookup", "sources": "referral_bonus", "answer_chars": 448}
{"@timestamp": "2026-11-06T21:18:07.203Z", "level": "INFO", "event": "request", "trace_id": "eae949d6a1384d5987847dfe6145389e", "session_id": "sess-3dd88e7a82eb", "endpoint": "/ask", "method": "POST", "status": "answered", "http_status": 200, "duration_ms": 46.146, "request_text": "What is the status of APP-0007?", "phone_numbers_masked": 0, "rejection_reason": null, "injection_matched": "", "resolution": "unchanged", "groundedness_action": "pass", "estimated_tokens": 608, "token_limit": 1000, "budget_allowed": true, "simulated_cost_usd": 0.001216, "cache_hits": 0, "cache_misses": 0, "review_approved": true, "review_revised": false, "tools": "check_job_application_status", "sources": "", "answer_chars": 149}
```

## Requests containing the phone number

- `/ask` (POST), trace_id `a3a9888974d34a5c83a623b6e418717a`: request_text = `Contact: [PHONE_MASKED]. How much referral bonus is paid for a managerial role?`; phone_numbers_masked = 1
- `/add-document` (POST), trace_id `b150cf490aa64bed84d139a617592c52`: request_text = `Front desk | 54 chars`; phone_numbers_masked = 1
- `/chat` (WEBSOCKET), trace_id `f50db376199a49bd96eb1862ebc8149e`: request_text = `Contact: [PHONE_MASKED]. What is the status of APP-0007?`; phone_numbers_masked = 1

## Checks on the log file bytes

- raw number `+91-98765-43210` present: False; digit fragments `98765` / `43210` present: False / False
- masked placeholder `[PHONE_MASKED]` occurrences: 2
- masking happens before the write: every string is passed through `mask_phone` in `render_line` before it is serialised.
- entries: 26; distinct trace IDs: 26
- Scope: only the fixed-format phone number is masked; names, expected salary and background-check text are not.
