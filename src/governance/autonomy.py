"""Application layer: least autonomy for the crew's tools.

Policy: each agent may hold only the tools listed here, and only the Lookup Agent may hold
`check_job_application_status`. `build_agents` calls `check_agent_tools` before constructing an agent, so wiring
the lookup tool to any other agent raises AutonomyViolation instead of producing a crew.
"""
LOOKUP_TOOL = "check_job_application_status"
ALLOWED_TOOLS = {
    "Retrieval Agent": frozenset({"rag_lookup"}),
    "Lookup Agent": frozenset({LOOKUP_TOOL}),
    "Response Composer": frozenset(),
}


class AutonomyViolation(RuntimeError):
    pass


def check_agent_tools(role, tools):
    if role not in ALLOWED_TOOLS:
        raise AutonomyViolation(f"unknown agent role {role!r}")
    names = {t.name for t in tools}
    extra = names - ALLOWED_TOOLS[role]
    if extra:
        raise AutonomyViolation(f"{role} may not hold {sorted(extra)}; allowed: {sorted(ALLOWED_TOOLS[role])}")
    if LOOKUP_TOOL in names and role != "Lookup Agent":
        raise AutonomyViolation(f"only the Lookup Agent may call {LOOKUP_TOOL}")


def tool_wiring(agents):
    """{role: [tool names]} for an iterable of crewai Agents (used by tests and transcripts)."""
    return {a.role: [t.name for t in a.tools] for a in agents}
