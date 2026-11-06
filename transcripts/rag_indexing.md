# RAG indexing

Embedding model: sentence-transformers/all-MiniLM-L6-v2 (local, offline). Knowledge-base documents: 12.

## fixed: collection `kb_fixed_overlap`
- chunks upserted: 40; collection count: 40
- sample query: 'How much referral bonus is paid for a managerial role?'
- top-1: referral_bonus::fixed::01 (similarity 0.7948): The bonus is INR 25,000 for non-managerial roles and INR 50,000 for managerial roles. A referral must be submitted through the portal before the candidate's first interview to qualify. Members

## sentence: collection `kb_sentence`
- chunks upserted: 48; collection count: 48
- sample query: 'How much referral bonus is paid for a managerial role?'
- top-1: referral_bonus::sent::00 (similarity 0.6964): An employee who refers an external candidate earns a referral bonus once the referred person joins and completes 90 days of employment.
