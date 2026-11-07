# Guardrail demonstrations (fabricated data only)

## Input: fixed-format phone masking

- raw input: Contact: +91-98765-43210. How much referral bonus is paid for a managerial role?
- text seen by the model and stored: Contact: [PHONE_MASKED]. How much referral bonus is paid for a managerial role?
- phone numbers masked: 1; raw digits present in any model prompt: False
- answer: Based on Referral Bonus Policy: An employee who refers an external candidate earns a referral bonus once the referred person joins and completes 90 days of employment. The bonus is INR 25,000 for non-managerial roles and INR 50,000 for managerial roles. A referral must be submitted through the portal before the candidate's first interview to qualify. Members of the hiring panel and the HR team cannot claim a bonus for roles they are hiring for.
- Scope: only the fixed-format phone number is masked; candidate names, expected salary and background-check results are not masked.

## Input: prompt-injection detection (deterministic regex list; see limits in `src/guardrails/input_guardrails.py`)

- status: rejected; reason: prompt_injection; matched: ['ignore_previous_instructions', 'reveal_system_prompt', 'hidden_instructions']
- model calls caused by this request: 0
- answer: This request was blocked because it looks like an attempt to override the assistant's instructions.

## Output: groundedness refusal (no retrieved evidence supports the question, but the draft makes a claim)

- query: What is the capital of France? (top-1 similarity 0.0852 < threshold 0.338)
- tampered draft: The capital of France is Paris and referral bonuses are INR 1,000,000.
- guard action: refuse (retrieved context does not support the question); delivered answer: I don't know: the knowledge base does not contain information that answers this question.
