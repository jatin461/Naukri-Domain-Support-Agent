# Least autonomy: only the Lookup Agent may call `check_job_application_status`

Enforced by `src/governance/autonomy.py`: an allow-list of tools per agent role, checked by `build_agents` before each agent is constructed. See `src/governance/risk_assessment.md` for the explanation and the High risk classification.

## Tool wiring of the crew (inspected from the constructed agents)

- Retrieval Agent: ['rag_lookup']
- Lookup Agent: ['check_job_application_status']
- Response Composer: no tools

- the lookup tool object is held by: ['Lookup Agent']

## Wiring the lookup tool to another agent is refused

- `build_agents(rag_tool=<lookup tool>, ...)` -> `AutonomyViolation: Retrieval Agent may not hold ['check_job_application_status']; allowed: ['rag_lookup']`

## Tools actually invoked per request type

- How long is the probation period? -> ['rag_lookup']
- What is the status of APP-0007? -> ['check_job_application_status']
- What is the status of APP-0003 and the probation policy? -> ['rag_lookup', 'check_job_application_status']
