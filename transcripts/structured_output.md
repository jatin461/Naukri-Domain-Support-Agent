# Structured crew output (`response_format = CrewResponse`, a Pydantic model passed as the Composer task's `output_pydantic` and re-validated in code)

- How long is the probation period?
  - validated: `{"answer":"Based on Probation Period Policy, Notice Period Policy: New hires serve a probation period of six months from the joining date. A review is held at the end of month three and again before month six, and the manager records whether performance targets are being met. Probation can be extended once, by up to three months, if a review shows The standard notice period is 60 days for confirmed employees and 30 days for employees who are still in probation. A reporting manager can approve an early release when the","response_type":"policy","sources":["probation_period","notice_period"],"grounded":true,"record_id":null,"status":null,"expected_salary_inr":null,"escalation_score":null,"escalation_recommended":null}`
- What is the status of APP-0007?
  - validated: `{"answer":"Application APP-0007 is at status Applied with expected salary INR 805,657. Escalation score 0.25 against threshold 0.33: escalation not recommended.","response_type":"lookup","sources":[],"grounded":false,"record_id":"APP-0007","status":"Applied","expected_salary_inr":805657,"escalation_score":0.2533,"escalation_recommended":false}`
- What is the capital of France?
  - validated: `{"answer":"I don't know: the knowledge base does not contain information that answers this question.","response_type":"no_evidence","sources":[],"grounded":false,"record_id":null,"status":null,"expected_salary_inr":null,"escalation_score":null,"escalation_recommended":null}`

## A malformed response is rejected

- `CrewResponse(status='Applied', no salary/score)` -> ValidationError: Value error, a looked-up record must carry record_id, expected_salary_inr, escalation_score and escalation_recommended
- crew returning that payload -> pipeline status `error`, response `None`, delivered answer: Sorry, I could not produce a validated answer for this request.
