# Tool invocation evidence (real crews driven by the mock CrewAI LLM)

## check_job_application_status: known and unknown IDs

- `check_job_application_status('APP-0007')` -> `{"found": true, "record_id": "APP-0007", "status": "Applied", "expected_salary_inr": 805657, "escalation_score": 0.2533, "escalation_threshold": 0.33, "escalation_recommended": false, "category": "Data Analyst", "days_since_created": 11, "flagged_priority_review": false}`
- `check_job_application_status('APP-9999')` -> `{"found": false, "record_id": "APP-9999", "error": "No application found for record ID APP-9999."}`

## Policy query: RAG tool expected, lookup tool not

- query: How much referral bonus is paid for a managerial role?
- crew run via `.kickoff()`; tools by agent: `{"Retrieval Agent": ["rag_lookup"], "Lookup Agent": ["check_job_application_status"], "Response Composer": []}`
- RAG tool invocations: `[{"tool": "rag_lookup", "args": {"query": "How much referral bonus is paid for a managerial role?"}}]`
- lookup tool invocations: `[]`
- validated response: `{"answer":"Based on Referral Bonus Policy: An employee who refers an external candidate earns a referral bonus once the referred person joins and completes 90 days of employment. The bonus is INR 25,000 for non-managerial roles and INR 50,000 for managerial roles. A referral must be submitted through the portal before the candidate's first interview to qualify. Members of the hiring panel and the HR team cannot claim a bonus for roles they are hiring for.","response_type":"policy","sources":["referral_bonus"],"grounded":true,"record_id":null,"status":null,"expected_salary_inr":null,"escalation_score":null,"escalation_recommended":null}`

## Status query: lookup tool expected, RAG tool not

- query: What is the status of application APP-0007?
- crew run via `.kickoff()`; tools by agent: `{"Retrieval Agent": ["rag_lookup"], "Lookup Agent": ["check_job_application_status"], "Response Composer": []}`
- RAG tool invocations: `[]`
- lookup tool invocations: `[{"tool": "check_job_application_status", "args": {"record_id": "APP-0007"}}]`
- validated response: `{"answer":"Application APP-0007 is at status Applied with expected salary INR 805,657. Escalation score 0.25 against threshold 0.33: escalation not recommended.","response_type":"lookup","sources":[],"grounded":false,"record_id":"APP-0007","status":"Applied","expected_salary_inr":805657,"escalation_score":0.2533,"escalation_recommended":false}`

## Unknown record ID

- query: What is the status of application APP-9999?
- crew run via `.kickoff()`; tools by agent: `{"Retrieval Agent": ["rag_lookup"], "Lookup Agent": ["check_job_application_status"], "Response Composer": []}`
- RAG tool invocations: `[]`
- lookup tool invocations: `[{"tool": "check_job_application_status", "args": {"record_id": "APP-9999"}}]`
- validated response: `{"answer":"No application found for record ID APP-9999.","response_type":"lookup","sources":[],"grounded":false,"record_id":"APP-9999","status":null,"expected_salary_inr":null,"escalation_score":null,"escalation_recommended":null}`

## Policy and record in one query: all three agents run

- query: What is the status of APP-0003 and the probation policy?
- crew run via `.kickoff()`; tools by agent: `{"Retrieval Agent": ["rag_lookup"], "Lookup Agent": ["check_job_application_status"], "Response Composer": []}`
- RAG tool invocations: `[{"tool": "rag_lookup", "args": {"query": "What is the status of APP-0003 and the probation policy?"}}]`
- lookup tool invocations: `[{"tool": "check_job_application_status", "args": {"record_id": "APP-0003"}}]`
- validated response: `{"answer":"Based on Probation Period Policy, Internal Transfer Eligibility, Notice Period Policy: … three and again before month six, and the manager records whether performance targets are being met. Probation can be extended once, by up to three months, if a review shows … has five working days to raise any business-critical concern. Employees who are still in probation are not eligible to transfer. … The standard notice period is 60 days for confirmed employees and 30 days for employees who are still in probation. A reporting manager can approve an early release when the … Application APP-0003 is at status Applied with expected salary INR 1,375,850. Escalation score 0.19 against threshold 0.33: escalation not recommended.","response_type":"policy_and_lookup","sources":["probation_period","internal_transfer","notice_period"],"grounded":true,"record_id":"APP-0003","status":"Applied","expected_salary_inr":1375850,"escalation_score":0.1867,"escalation_recommended":false}`
