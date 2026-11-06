# Runtime budget: a deliberately oversized request is rejected

Estimate (deterministic, simulated): `ceil(len(masked question) / 4) + 600`; limit 1,000 tokens; simulated cost = tokens / 1000 x USD 0.002 (illustrative). No provider is called in `MOCK_LLM` mode, so nothing is actually charged.

## A normal request

- HTTP 200; status answered; budget `{"allowed": true, "estimated_tokens": 609, "limit": 1000, "simulated_cost_usd": 0.001218}`

## The oversized request (2,000 words, 10,000 characters)

- HTTP 413; status rejected; rejection_reason budget_exceeded
- budget `{"allowed": false, "estimated_tokens": 3100, "limit": 1000, "simulated_cost_usd": 0.0062}`
- answer: Request rejected: estimated 3100 tokens exceeds the per-request limit of 1000.
- model calls caused by the rejected request: 0 (rejected before the crew is built); sessions created by the rejected request: 0 (no session history is stored)
- boundary: the largest accepted question is 1600 characters (estimate exactly 1,000); one more character is rejected (`check_budget` allowed: True / False)
