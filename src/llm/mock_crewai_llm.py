"""Thin CrewAI adapter: a crewai BaseLLM subclass that delegates to the shared MockLLM core."""
from typing import Any

from crewai.llms.base_llm import BaseLLM
from pydantic import Field

from src.llm.mock_llm import MockLLM, ToolSpec, get_mock_llm


def _tool_specs(from_agent):
    # With a plain BaseLLM, CrewAI passes tools=None to call(); the declared tools live on the agent.
    specs = []
    for t in getattr(from_agent, "tools", None) or []:
        props = t.args_schema.model_json_schema().get("properties", {}) if getattr(t, "args_schema", None) else {}
        specs.append(ToolSpec(t.name, getattr(t, "description", ""), props))
    return specs


class MockCrewAILLM(BaseLLM):
    llm_type: str = "mock"
    core: Any = Field(default_factory=get_mock_llm, exclude=True)

    def call(self, messages, tools=None, callbacks=None, available_functions=None, from_task=None, from_agent=None, response_model=None):
        return self.core.complete("crew", messages, _tool_specs(from_agent), response_model)

    def supports_stop_words(self) -> bool:
        return False


def make_mock_crewai_llm(core: MockLLM | None = None):
    return MockCrewAILLM(model="mock-llm", **({"core": core} if core else {}))
