"""Thin AutoGen adapter: a ChatCompletionClient that delegates to the shared MockLLM core (purpose "review")."""
from autogen_core.models import ChatCompletionClient, CreateResult, ModelInfo, RequestUsage

from src.llm.mock_llm import MockLLM, get_mock_llm


def _to_chat(messages):
    out = []
    for m in messages:
        kind = type(m).__name__
        role = {"SystemMessage": "system", "UserMessage": "user", "AssistantMessage": "assistant"}.get(kind, "user")
        content = m.content if isinstance(m.content, str) else str(m.content)
        out.append({"role": role, "name": getattr(m, "source", None), "content": content})
    return out


class MockAutoGenClient(ChatCompletionClient):
    def __init__(self, core: MockLLM | None = None):
        self.core = core or get_mock_llm()
        self._usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    @property
    def model_info(self):
        return ModelInfo(vision=False, function_calling=False, json_output=True, family="unknown", structured_output=True)

    @property
    def capabilities(self):
        return self.model_info

    async def create(self, messages, *, tools=[], tool_choice="auto", json_output=None, extra_create_args={}, cancellation_token=None):
        text = self.core.complete("review", _to_chat(messages))
        return CreateResult(finish_reason="stop", content=text, usage=self._usage, cached=False)

    async def create_stream(self, *args, **kwargs):
        raise NotImplementedError("the mock review client does not stream")

    async def close(self):
        pass

    def actual_usage(self):
        return self._usage

    def total_usage(self):
        return self._usage

    def count_tokens(self, messages, *, tools=[]):
        return 0

    def remaining_tokens(self, messages, *, tools=[]):
        return 10_000
