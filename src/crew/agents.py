"""The three crew agents. All are always constructed with fixed tool wiring (least autonomy):
Retrieval Agent: RAG tool only; Lookup Agent: application-lookup tool only; Composer: no tools.
Delegation is off, crew memory is off, and the mock LLM is injected into every LLM slot."""
from typing import NamedTuple

from crewai import Agent

from src.governance.autonomy import check_agent_tools


class Agents(NamedTuple):
    retrieval: Agent
    lookup: Agent
    composer: Agent


def build_agents(llm, rag_tool, lookup_tool):
    def make(role, goal, backstory, tools):
        check_agent_tools(role, tools)  # least autonomy: refuse a forbidden tool wiring before the agent exists
        return Agent(role=role, goal=goal, backstory=backstory, tools=tools, llm=llm, allow_delegation=False, verbose=False)

    return Agents(
        retrieval=make("Retrieval Agent", "Retrieve policy evidence from the knowledge base.",
                       "You answer hiring-policy questions only from the knowledge base.", [rag_tool]),
        lookup=make("Lookup Agent", "Look up the requested job application record.",
                    "You read application status, expected salary and escalation score from the dataset.", [lookup_tool]),
        composer=make("Response Composer", "Combine the gathered evidence into one structured draft response.",
                      "You write the final draft strictly from the evidence handed to you.", []),
    )
