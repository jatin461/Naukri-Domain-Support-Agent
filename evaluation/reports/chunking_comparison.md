# Chunking strategy comparison (document-level precision and recall)

Top-k = 3 chunks per query; retrieved chunks are mapped to parent documents and deduplicated before scoring.
Precision = relevant documents retrieved / distinct documents retrieved. Recall = relevant documents retrieved / relevant documents declared.
Fixed strategy: 30-word windows, 8-word overlap. Sentence strategy: one chunk per sentence.

## kb_fixed_overlap (fixed), 40 chunks

| # | Query | Relevant (declared) | Retrieved chunks | Retrieved docs (deduplicated) | Arithmetic |
|---|---|---|---|---|---|
| 1 | How many hours before an interview can I reschedule my slot? | interview_scheduling | interview_scheduling::fixed::02, interview_scheduling::fixed::01, interview_scheduling::fixed::00 | interview_scheduling | P = 1/1 = 1.000; R = 1/1 = 1.000 |
| 2 | What is the notice period for an employee who is still in probation? | notice_period, probation_period | notice_period::fixed::00, probation_period::fixed::01, probation_period::fixed::00 | notice_period, probation_period | P = 2/2 = 1.000; R = 2/2 = 1.000 |
| 3 | How much referral bonus is paid for a managerial role? | referral_bonus | referral_bonus::fixed::01, referral_bonus::fixed::00, referral_bonus::fixed::02 | referral_bonus | P = 1/1 = 1.000; R = 1/1 = 1.000 |
| 4 | Who can apply for an internal transfer and how long must they have been in their role? | internal_transfer | internal_transfer::fixed::00, job_application_eligibility::fixed::01, job_application_eligibility::fixed::00 | internal_transfer, job_application_eligibility | P = 1/2 = 0.500; R = 1/1 = 1.000 |
| 5 | How long do you keep the data of rejected applicants? | applicant_data_retention | applicant_data_retention::fixed::00, applicant_data_retention::fixed::02, job_application_eligibility::fixed::03 | applicant_data_retention, job_application_eligibility | P = 1/2 = 0.500; R = 1/1 = 1.000 |
| 6 | Can a new joiner work from home during probation? | probation_period, remote_work | remote_work::fixed::00, internal_transfer::fixed::02, probation_period::fixed::00 | remote_work, internal_transfer, probation_period | P = 2/3 = 0.667; R = 2/2 = 1.000 |

Mean precision = 0.778; mean recall = 1.000 over 6 queries.

## kb_sentence (sentence), 48 chunks

| # | Query | Relevant (declared) | Retrieved chunks | Retrieved docs (deduplicated) | Arithmetic |
|---|---|---|---|---|---|
| 1 | How many hours before an interview can I reschedule my slot? | interview_scheduling | interview_scheduling::sent::02, interview_scheduling::sent::00, interview_scheduling::sent::03 | interview_scheduling | P = 1/1 = 1.000; R = 1/1 = 1.000 |
| 2 | What is the notice period for an employee who is still in probation? | notice_period, probation_period | notice_period::sent::00, probation_period::sent::00, probation_period::sent::02 | notice_period, probation_period | P = 2/2 = 1.000; R = 2/2 = 1.000 |
| 3 | How much referral bonus is paid for a managerial role? | referral_bonus | referral_bonus::sent::00, referral_bonus::sent::01, referral_bonus::sent::03 | referral_bonus | P = 1/1 = 1.000; R = 1/1 = 1.000 |
| 4 | Who can apply for an internal transfer and how long must they have been in their role? | internal_transfer | internal_transfer::sent::00, internal_transfer::sent::03, job_application_eligibility::sent::01 | internal_transfer, job_application_eligibility | P = 1/2 = 0.500; R = 1/1 = 1.000 |
| 5 | How long do you keep the data of rejected applicants? | applicant_data_retention | applicant_data_retention::sent::00, job_application_eligibility::sent::02, applicant_data_retention::sent::01 | applicant_data_retention, job_application_eligibility | P = 1/2 = 0.500; R = 1/1 = 1.000 |
| 6 | Can a new joiner work from home during probation? | probation_period, remote_work | remote_work::sent::00, internal_transfer::sent::03, probation_period::sent::00 | remote_work, internal_transfer, probation_period | P = 2/3 = 0.667; R = 2/2 = 1.000 |

Mean precision = 0.778; mean recall = 1.000 over 6 queries.

## Summary

| Strategy | Mean precision | Mean recall |
|---|---|---|
| fixed-size with overlap | 0.778 | 1.000 |
| sentence-based | 0.778 | 1.000 |

## Sensitivity to top-k (same queries; primary result above uses k = 3)

| k | fixed precision | fixed recall | sentence precision | sentence recall |
|---|---|---|---|---|
| 1 | 1.000 | 0.833 | 1.000 | 0.833 |
| 2 | 0.833 | 0.917 | 0.833 | 0.917 |
| 3 | 0.778 | 1.000 | 0.778 | 1.000 |
| 5 | 0.444 | 1.000 | 0.444 | 1.000 |

## Harder multi-document set (each query spans three documents)

The six shared queries above are the Task 4 queries and tie. These three extra queries each need evidence from three documents, so with top-k = 3 full recall requires every retrieved chunk to come from a different relevant document.

### kb_fixed_overlap [fixed], harder set

| # | Query | Relevant (declared) | Retrieved chunks | Retrieved docs (deduplicated) | Arithmetic |
|---|---|---|---|---|---|
| H1 | If I resign while still on probation, how much notice do I give and will HR hold an exit interview? | exit_interview, notice_period, probation_period | exit_interview::fixed::00, notice_period::fixed::00, probation_period::fixed::00 | exit_interview, notice_period, probation_period | P = 3/3 = 1.000; R = 3/3 = 1.000 |
| H2 | Once my probation is confirmed, can I work remotely and also move to another team internally? | internal_transfer, probation_period, remote_work | remote_work::fixed::00, internal_transfer::fixed::02, remote_work::fixed::01 | remote_work, internal_transfer | P = 2/2 = 1.000; R = 2/3 = 0.667 |
| H3 | Does a new hire on probation get remote work, and what notice applies if they leave? | notice_period, probation_period, remote_work | remote_work::fixed::00, probation_period::fixed::00, internal_transfer::fixed::02 | remote_work, probation_period, internal_transfer | P = 2/3 = 0.667; R = 2/3 = 0.667 |

Mean precision = 0.889; mean recall = 0.778 over 3 queries.

### kb_sentence [sentence], harder set

| # | Query | Relevant (declared) | Retrieved chunks | Retrieved docs (deduplicated) | Arithmetic |
|---|---|---|---|---|---|
| H1 | If I resign while still on probation, how much notice do I give and will HR hold an exit interview? | exit_interview, notice_period, probation_period | exit_interview::sent::00, notice_period::sent::00, probation_period::sent::00 | exit_interview, notice_period, probation_period | P = 3/3 = 1.000; R = 3/3 = 1.000 |
| H2 | Once my probation is confirmed, can I work remotely and also move to another team internally? | internal_transfer, probation_period, remote_work | remote_work::sent::00, internal_transfer::sent::03, probation_period::sent::02 | remote_work, internal_transfer, probation_period | P = 3/3 = 1.000; R = 3/3 = 1.000 |
| H3 | Does a new hire on probation get remote work, and what notice applies if they leave? | notice_period, probation_period, remote_work | remote_work::sent::00, notice_period::sent::00, probation_period::sent::00 | remote_work, notice_period, probation_period | P = 3/3 = 1.000; R = 3/3 = 1.000 |

Mean precision = 1.000; mean recall = 1.000 over 3 queries.

### All results side by side

| Query set | fixed precision | fixed recall | sentence precision | sentence recall |
|---|---|---|---|---|
| 6 shared queries | 0.778 | 1.000 | 0.778 | 1.000 |
| 3 harder queries | 0.889 | 0.778 | 1.000 | 1.000 |
| all 9 queries | 0.815 | 0.926 | 0.852 | 1.000 |

Harder set, sensitivity to top-k:

| k | fixed precision | fixed recall | sentence precision | sentence recall |
|---|---|---|---|---|
| 1 | 1.000 | 0.333 | 1.000 | 0.333 |
| 2 | 1.000 | 0.667 | 1.000 | 0.667 |
| 3 | 0.889 | 0.778 | 1.000 | 1.000 |
| 5 | 0.833 | 1.000 | 0.917 | 1.000 |

## Recommendation (hand-written)

On the six shared Task 4 queries the two collections tie at k = 3 (mean precision 0.778 and mean recall 1.000 for both), but on the three harder queries that each span three documents sentence-based chunking pulls ahead with mean precision 1.000 and mean recall 1.000 against 0.889 and 0.778 for fixed-size, because the 30-word windows spent a top-3 slot on a second remote-work window in one query and on the neighbouring internal-transfer document instead of the notice-period document in another (fixed-size only reaches recall 1.000 at k = 5). Retrieval alone therefore favours sentence-based chunking, but in the 15-query end-to-end evaluation the fixed-size collection scored completeness 5.000 against 4.733, because a single retrieved sentence often leaves out the second fact a policy answer needs, and most real questions to this assistant are about a single topic. I therefore recommend deploying the fixed-size collection for this knowledge base, accepting weaker coverage on three-topic questions and a narrower fallback margin (the dress-code query scores 0.3220 against a threshold of 0.338, a gap of 0.0160, versus 0.1019 on the sentence-based collection at 0.354), and I would switch to sentence-based chunking if multi-topic questions became common.
