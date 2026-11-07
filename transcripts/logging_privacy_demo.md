# Logging and privacy demonstration

Log file: `logs/sample_requests.jsonl` (26 entries: every request of the API demo plus three requests that contain the fabricated phone number +91-98765-43210).

## One flat JSON object per request (ELK style)

```json
{"@timestamp": "2026-11-07T13:29:20.975Z", "level": "INFO", "event": "request", "trace_id": "40ed3e13804943ad9cb8fd39e909e420", "session_id": "sess-e87d1742aa6a", "endpoint": "/ask", "method": "POST", "status": "answered", "http_status": 200, "duration_ms": 51.483, "request_text": "How much referral bonus is paid for a managerial role?", "phone_numbers_masked": 0, "rejection_reason": null, "injection_matched": "", "resolution": "unchanged", "groundedness_action": "pass", "estimated_tokens": 614, "token_limit": 1000, "budget_allowed": true, "simulated_cost_usd": 0.001228, "cache_hits": 0, "cache_misses": 1, "review_approved": true, "review_revised": false, "tools": "rag_lookup", "sources": "referral_bonus", "answer_chars": 448}
{"@timestamp": "2026-11-07T13:29:21.019Z", "level": "INFO", "event": "request", "trace_id": "ce993467fe484adb975849590b488bea", "session_id": "sess-8b12ad6bc179", "endpoint": "/ask", "method": "POST", "status": "answered", "http_status": 200, "duration_ms": 41.219, "request_text": "What is the status of APP-0007?", "phone_numbers_masked": 0, "rejection_reason": null, "injection_matched": "", "resolution": "unchanged", "groundedness_action": "pass", "estimated_tokens": 608, "token_limit": 1000, "budget_allowed": true, "simulated_cost_usd": 0.001216, "cache_hits": 0, "cache_misses": 0, "review_approved": true, "review_revised": false, "tools": "check_job_application_status", "sources": "", "answer_chars": 149}
```

## Requests containing the phone number

- `/ask` (POST), trace_id `65b7e9fbd49b4bf8a31985fc49802bdf`: request_text = `Contact: [PHONE_MASKED]. How much referral bonus is paid for a managerial role?`; phone_numbers_masked = 1
- `/add-document` (POST), trace_id `a8fbc9d60f4e43f0963acd80716878f0`: request_text = `Front desk | 54 chars`; phone_numbers_masked = 1
- `/chat` (WEBSOCKET), trace_id `67f0b977e57342b19e7bff49bddaa82a`: request_text = `Contact: [PHONE_MASKED]. What is the status of APP-0007?`; phone_numbers_masked = 1

## Checks on the log file bytes

- raw number `+91-98765-43210` present: False; digit fragments `98765` / `43210` present: False / False
- masked placeholder `[PHONE_MASKED]` occurrences: 2
- masking happens before the write: every string is passed through `mask_phone` in `render_line` before it is serialised.
- entries: 26; distinct trace IDs: 26
- Scope: only the fixed-format phone number is masked; names, expected salary and background-check text are not.
