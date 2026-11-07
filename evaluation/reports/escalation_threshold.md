# Escalation score and threshold

Formula: `escalation_score = 0.60 * flagged_priority_review + 0.40 * (1 - days_since_created / 30)`; recommended escalation when score >= **0.33**.
Newer applications score higher (they are still inside their first-response window, while older ones have already been through recruiter triage); the priority flag adds a fixed 0.60.

Computed from the generated dataset (60 records, seed 42):

- days_since_created percentiles (linear interpolation): P20 = 5.80, P50 = 12.00, P80 = 23.00
- score range over all records: 0.0133 to 1.0000 (inside [0, 1])
- flagged records score at least 0.6800; unflagged records at most 0.3867
- unflagged records escalate only when days_since_created <= 5 (every such record is below P20 = 5.80); 10 unflagged records qualify
- escalated: 18 of 60 = 8 flagged + 10 unflagged-but-recent

Decision regions at the chosen threshold:

| days_since_created | unflagged score | unflagged escalates | flagged score | flagged escalates |
|---|---|---|---|---|
| 0 | 0.4000 | yes | 1.0000 | yes |
| 1 | 0.3867 | yes | 0.9867 | yes |
| 2 | 0.3733 | yes | 0.9733 | yes |
| 3 | 0.3600 | yes | 0.9600 | yes |
| 4 | 0.3467 | yes | 0.9467 | yes |
| 5 | 0.3333 | yes | 0.9333 | yes |
| 6 | 0.3200 | no | 0.9200 | yes |
| 7 | 0.3067 | no | 0.9067 | yes |
| 10 | 0.2667 | no | 0.8667 | yes |
| 15 | 0.2000 | no | 0.8000 | yes |
| 20 | 0.1333 | no | 0.7333 | yes |
| 23 | 0.0933 | no | 0.6933 | yes |
| 25 | 0.0667 | no | 0.6667 | yes |
| 30 | 0.0000 | no | 0.6000 | yes |

Any threshold in (0.40, 0.60] would equal the boolean flag exactly, so the threshold is deliberately below 0.40, where both components change outcomes.

## Justification (hand-written)

The 20th percentile of days_since_created in the generated dataset is 5.80 and the 80th is 23.00, so I set the threshold at 0.33, which sits just below the score of an unflagged application created 5 days ago (0.3333) and above that of one created 6 days ago (0.3200). That makes an unflagged application escalate only when it is fresher than the dataset's P20, which is 10 of the 60 records, while all 8 flagged records escalate because their lowest score is 0.6800; together that is 18 of 60 escalations, 8 driven by the flag and 10 by recency alone. The threshold sits below 0.40 on purpose, because any value between 0.40 and 0.60 would reproduce the boolean flag exactly, and the P80 of 23 days shows why the flag still matters: an unflagged application that old scores only 0.0933, while a flagged one scores 0.6933. The score only routes a case to a human reviewer; it is not a judgement about the candidate.
