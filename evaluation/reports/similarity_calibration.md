# Similarity calibration

Metric: top-1 cosine similarity (1 - Chroma cosine distance), model `sentence-transformers/all-MiniLM-L6-v2`, top-k chunks retrieved per query.
Under `MOCK_LLM` this similarity is the only groundedness signal; no model judges semantic groundedness.

## Collection: kb_fixed_overlap (fixed)

| Set | Query | Top-1 similarity | Top-1 doc |
|---|---|---|---|
| in-scope | How many hours before an interview can I reschedule my slot? | 0.7099 | interview_scheduling |
| in-scope | What is the notice period for an employee who is still in probation? | 0.7917 | notice_period |
| in-scope | How much referral bonus is paid for a managerial role? | 0.7948 | referral_bonus |
| in-scope | Who can apply for an internal transfer and how long must they have been in their role? | 0.7725 | internal_transfer |
| in-scope | How long do you keep the data of rejected applicants? | 0.7839 | applicant_data_retention |
| in-scope | Can a new joiner work from home during probation? | 0.5527 | remote_work |
| out-of-scope | What is the capital of France? | 0.0852 | job_application_eligibility |
| out-of-scope | How do I bake a chocolate cake? | 0.0458 | job_application_eligibility |
| out-of-scope | Who won the football world cup in 2018? | 0.1152 | internal_transfer |
| out-of-scope | Which programming language is best for web scraping? | 0.1235 | diversity_hiring |

- In-scope cluster: min 0.5527, max 0.7948
- Out-of-scope cluster: min 0.0458, max 0.1235
- Gap between clusters: 0.4292
- **Chosen threshold (midpoint of gap): 0.338**

- Closest measured query to the threshold (all 12 queries): 0.0160 away ("What is the company dress code for the office?")

Held-out check (not used to choose the threshold):

| Set | Query | Top-1 similarity | Behaves correctly at threshold |
|---|---|---|---|
| in-scope | What happens if my background check shows a mismatch? | 0.5453 | yes |
| in-scope | Is a bonus given for referring a friend to the company? | 0.6200 | yes |
| out-of-scope | What is the weather in Mumbai today? | 0.1001 | yes |
| out-of-scope | What is the company dress code for the office? | 0.3220 | yes |

## Collection: kb_sentence (sentence)

| Set | Query | Top-1 similarity | Top-1 doc |
|---|---|---|---|
| in-scope | How many hours before an interview can I reschedule my slot? | 0.6938 | interview_scheduling |
| in-scope | What is the notice period for an employee who is still in probation? | 0.8533 | notice_period |
| in-scope | How much referral bonus is paid for a managerial role? | 0.6964 | referral_bonus |
| in-scope | Who can apply for an internal transfer and how long must they have been in their role? | 0.7961 | internal_transfer |
| in-scope | How long do you keep the data of rejected applicants? | 0.7738 | applicant_data_retention |
| in-scope | Can a new joiner work from home during probation? | 0.5569 | remote_work |
| out-of-scope | What is the capital of France? | 0.0238 | job_application_eligibility |
| out-of-scope | How do I bake a chocolate cake? | 0.0767 | referral_bonus |
| out-of-scope | Who won the football world cup in 2018? | 0.0809 | probation_period |
| out-of-scope | Which programming language is best for web scraping? | 0.1509 | diversity_hiring |

- In-scope cluster: min 0.5569, max 0.8533
- Out-of-scope cluster: min 0.0238, max 0.1509
- Gap between clusters: 0.4060
- **Chosen threshold (midpoint of gap): 0.354**

- Closest measured query to the threshold (all 12 queries): 0.1019 away ("What is the company dress code for the office?")

Held-out check (not used to choose the threshold):

| Set | Query | Top-1 similarity | Behaves correctly at threshold |
|---|---|---|---|
| in-scope | What happens if my background check shows a mismatch? | 0.4663 | yes |
| in-scope | Is a bonus given for referring a friend to the company? | 0.6222 | yes |
| out-of-scope | What is the weather in Mumbai today? | 0.1125 | yes |
| out-of-scope | What is the company dress code for the office? | 0.2521 | yes |

## Rationale (hand-written)

The two clusters never touch. On the fixed-size collection every in-scope query scored at least 0.5527 and every out-of-scope query at most 0.1235, a gap of 0.4292, so the threshold 0.338 is the midpoint of that gap; the sentence-based collection gave a gap of 0.4060 and a midpoint of 0.354. A round preset would have been wrong in practice: 0.6 would refuse the in-scope remote-work query (0.5527) and the held-out background-check query (0.5453). The HR-adjacent dress-code query (0.3220) is correctly refused but only 0.0160 below the threshold, so questions that sit close to HR topics without being in the knowledge base are the likeliest false accepts, and ten calibration queries are a small sample for a topical-overlap proxy that does not prove an answer is supported.
