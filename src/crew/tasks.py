"""Per-request task list. Task descriptions end with `Question: <text>`."""
from crewai import Task

from src.schemas.crew_response import response_format


def build_tasks(agents, question, route):
    gather = []
    if route.needs_policy:
        gather.append(Task(description=f"Retrieve policy evidence from the knowledge base.\nQuestion: {question}",
                           expected_output="Grounded policy evidence with source documents.", agent=agents.retrieval))
    if route.needs_lookup:
        gather.append(Task(description=f"Look up the application record.\nQuestion: {question}",
                           expected_output="Application status, expected salary and escalation score.", agent=agents.lookup))
    compose = Task(description="Compose the final draft response from the gathered evidence.",
                   expected_output="A response matching the CrewResponse schema.", agent=agents.composer,
                   context=gather, output_pydantic=response_format)
    return [*gather, compose]
