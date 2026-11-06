# Session memory

## Session A (multi-turn: later turns use earlier context)

- [session-A] User: What is the status of APP-0007?
  - status: answered; resolution: unchanged; tools: check_job_application_status({"record_id": "APP-0007"})
  - answer: Application APP-0007 is at status Applied with expected salary INR 805,657. Escalation score 0.25 against threshold 0.33: escalation not recommended.
- [session-A] User: and what is its expected salary?
  - status: answered; resolution: resolved_record; tools: check_job_application_status({"record_id": "APP-0007"})
  - answer: Application APP-0007 is at status Applied with expected salary INR 805,657. Escalation score 0.25 against threshold 0.33: escalation not recommended.
- [session-A] User: How much referral bonus is paid for a managerial role?
  - status: answered; resolution: unchanged; tools: rag_lookup({"query": "How much referral bonus is paid for a managerial role?"})
  - answer: Based on Referral Bonus Policy: An employee who refers an external candidate earns a referral bonus once the referred person joins and completes 90 days of employment. The bonus is INR 25,000 for non-managerial roles and INR 50,000 for managerial roles. A referral must be submitted through the portal before the candidate's first interview to qualify. Members of the hiring panel and the HR team cannot claim a bonus for roles they are hiring for.
- [session-A] User: And for non-managerial roles?
  - status: answered; resolution: resolved_follow_up; tools: rag_lookup({"query": "How much referral bonus is paid for a managerial role? Follow-up: And for non-managerial roles?"})
  - answer: Based on Referral Bonus Policy: An employee who refers an external candidate earns a referral bonus once the referred person joins and completes 90 days of employment. The bonus is INR 25,000 for non-managerial roles and INR 50,000 for managerial roles. A referral must be submitted through the portal before the candidate's first interview to qualify. Members of the hiring panel and the HR team cannot claim a bonus for roles they are hiring for.

## Session B (fresh session: no state from session A)

- [session-B] User: and what is its expected salary?
  - status: clarification; resolution: clarify; tools: none
  - answer: Which application do you mean? Please include a record ID such as APP-0007.

Crew runs for that turn: 0 (clarification issued before the crew). History messages: A = 8, B = 2.

Limits: the resolver assumes the most recent record or policy question in the session is meant; it understands only the cue words in `src/crew/routing.py`.
