"""Query sets for threshold calibration and chunking comparison.

Relevant parent documents are declared here, before any retrieval is scored.
IN_SCOPE are the queries shared by Task 4 (calibration, demos) and Task 5 (precision/recall).
HELD_OUT queries are not used to pick the threshold; they only check it afterwards.
"""

IN_SCOPE = [
    {"query": "How many hours before an interview can I reschedule my slot?", "relevant": ["interview_scheduling"]},
    {
        "query": "What is the notice period for an employee who is still in probation?",
        "relevant": ["notice_period", "probation_period"],
    },
    {"query": "How much referral bonus is paid for a managerial role?", "relevant": ["referral_bonus"]},
    {
        "query": "Who can apply for an internal transfer and how long must they have been in their role?",
        "relevant": ["internal_transfer"],
    },
    {"query": "How long do you keep the data of rejected applicants?", "relevant": ["applicant_data_retention"]},
    {
        "query": "Can a new joiner work from home during probation?",
        "relevant": ["remote_work", "probation_period"],
    },
]

OUT_OF_SCOPE = [
    "What is the capital of France?",
    "How do I bake a chocolate cake?",
    "Who won the football world cup in 2018?",
    "Which programming language is best for web scraping?",
]

HELD_OUT_IN_SCOPE = [
    {"query": "What happens if my background check shows a mismatch?", "relevant": ["background_verification"]},
    {"query": "Is a bonus given for referring a friend to the company?", "relevant": ["referral_bonus"]},
]
HELD_OUT_OUT_OF_SCOPE = [
    "What is the weather in Mumbai today?",
    "What is the company dress code for the office?",
]

# Harder Task 5 set: every query spans THREE knowledge-base documents, so a top-3 result can only reach full recall if each
# retrieved chunk comes from a different relevant document. Used only for the chunking comparison (not for calibration).
HARD_MULTI_DOC = [
    {
        "query": "If I resign while still on probation, how much notice do I give and will HR hold an exit interview?",
        "relevant": ["notice_period", "probation_period", "exit_interview"],
    },
    {
        "query": "Once my probation is confirmed, can I work remotely and also move to another team internally?",
        "relevant": ["probation_period", "remote_work", "internal_transfer"],
    },
    {
        "query": "Does a new hire on probation get remote work, and what notice applies if they leave?",
        "relevant": ["probation_period", "remote_work", "notice_period"],
    },
]
