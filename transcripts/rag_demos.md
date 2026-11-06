# Grounded generation demos

Collection: `kb_fixed_overlap`; calibrated fallback threshold: 0.338; top-k: 3.
Groundedness signal under MOCK_LLM is top-1 retrieval similarity only.

## In-scope queries (answered from retrieved context)

- Query: How many hours before an interview can I reschedule my slot?
  - top-1 similarity: 0.7099 (threshold 0.338) -> grounded=True
  - sources: interview_scheduling
  - answer: Based on Interview Scheduling Process: When an application reaches Screening, a recruiter proposes at least two interview slots within the next five working days. The candidate confirms one slot through the portal, and the status changes to Interview Scheduled only after that confirmation. A candidate may reschedule once without penalty if the request is made at least 24 hours before the slot. Each interview round is limited to 60 minutes, and the panel must submit written feedback within two
- Query: What is the notice period for an employee who is still in probation?
  - top-1 similarity: 0.7917 (threshold 0.338) -> grounded=True
  - sources: notice_period, probation_period
  - answer: Based on Notice Period Policy, Probation Period Policy: The standard notice period is 60 days for confirmed employees and 30 days for employees who are still in probation. A reporting manager can approve an early release when the New hires serve a probation period of six months from the joining date. A review is held at the end of month three and again before month six, and the manager records whether performance targets are being met. Probation can be extended once, by up to three months, if a review shows
- Query: How much referral bonus is paid for a managerial role?
  - top-1 similarity: 0.7948 (threshold 0.338) -> grounded=True
  - sources: referral_bonus
  - answer: Based on Referral Bonus Policy: An employee who refers an external candidate earns a referral bonus once the referred person joins and completes 90 days of employment. The bonus is INR 25,000 for non-managerial roles and INR 50,000 for managerial roles. A referral must be submitted through the portal before the candidate's first interview to qualify. Members of the hiring panel and the HR team cannot claim a bonus for roles they are hiring for.
- Query: Who can apply for an internal transfer and how long must they have been in their role?
  - top-1 similarity: 0.7725 (threshold 0.338) -> grounded=True
  - sources: internal_transfer, job_application_eligibility
  - answer: Based on Internal Transfer Eligibility, Job Application Eligibility: Employees may apply for an internal transfer after completing 12 months in their current role. They need a rating of Meets Expectations or higher in the latest performance review cycle. An applicant is eligible for a posted role when the resume shows the minimum years of experience stated in the job description and any mandatory qualification listed for that role. Applicants must be at least 18 years old and legally permitted to work in the country where the position is based. A
- Query: How long do you keep the data of rejected applicants?
  - top-1 similarity: 0.7839 (threshold 0.338) -> grounded=True
  - sources: applicant_data_retention, job_application_eligibility
  - answer: Based on Applicant Data Retention Policy, Job Application Eligibility: Data of unsuccessful applicants is kept for 12 months after the final decision, after which it is deleted or anonymised. Applicants who are hired have their data moved to the writing for earlier deletion, and HR answers within 30 days. Access to application data is limited to the recruiters and hiring managers involved in that role. time, and a rejected candidate may reapply for the same role after 90 days. Applications without a current contact email stay at Applied until that field is completed.
- Query: Can a new joiner work from home during probation?
  - top-1 similarity: 0.5527 (threshold 0.338) -> grounded=True
  - sources: remote_work, internal_transfer, probation_period
  - answer: Based on Remote Work Eligibility, Internal Transfer Eligibility, Probation Period Policy: Employees become eligible for remote work once their employment has been confirmed after probation. Roles that need on-site equipment or in-person customer contact are excluded from the arrangement. Eligible employees has five working days to raise any business-critical concern. Employees who are still in probation are not eligible to transfer. New hires serve a probation period of six months from the joining date. A review is held at the end of month three and again before month six, and the

## Out-of-scope queries (fallback)

- Query: What is the capital of France?
  - top-1 similarity: 0.0852 (threshold 0.338) -> grounded=False
  - answer: I don't know: the knowledge base does not contain information that answers this question.
- Query: What is the company dress code for the office?
  - top-1 similarity: 0.3220 (threshold 0.338) -> grounded=False
  - answer: I don't know: the knowledge base does not contain information that answers this question.
