"""Runtime layer: per-request token/cost budget (simulated).

Estimate (deterministic): estimated_tokens = ceil(len(masked_question) / 4) + PROMPT_OVERHEAD_TOKENS, where
the overhead is a fixed allowance for the agent prompts, retrieved context and review prompt. A request whose
estimate exceeds MAX_REQUEST_TOKENS is rejected before the crew is built. The simulated cost is
estimated_tokens / 1000 * SIMULATED_USD_PER_1K_TOKENS. In MOCK_LLM mode no provider is called, so nothing is
actually charged; the budget is a simulated operational control that would cap real spend if a real LLM were wired in.
"""
import math
from dataclasses import asdict, dataclass

from src.config import settings


def estimate_tokens(text):
    return math.ceil(len(text) / 4)


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    estimated_tokens: int
    limit: int
    simulated_cost_usd: float

    def as_dict(self):
        return asdict(self)


def check_budget(masked_question, limit=None):
    limit = settings.MAX_REQUEST_TOKENS if limit is None else limit
    est = estimate_tokens(masked_question) + settings.PROMPT_OVERHEAD_TOKENS
    cost = round(est / 1000 * settings.SIMULATED_USD_PER_1K_TOKENS, 6)
    return BudgetDecision(est <= limit, est, limit, cost)
