import json
from dataclasses import dataclass
from pathlib import Path

from efficiency_gateway.core.token_counter import TokenCounter


@dataclass(frozen=True)
class CostEstimate:
    model: str
    provider: str
    currency: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    tokenizer: str


class CostTracker:
    def __init__(self, pricing_path: str = "configs/pricing.json"):
        self.pricing_path = Path(pricing_path)
        self.catalog = self._load_pricing()
        self.token_counter = TokenCounter()

    def _load_pricing(self):
        with self.pricing_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def estimate_from_text(self, model: str, input_text: str, output_text: str = ""):
        input_count = self.token_counter.count(input_text)
        output_count = self.token_counter.count(output_text)

        return self.estimate_from_counts(
            model=model,
            input_tokens=input_count.tokens,
            output_tokens=output_count.tokens,
            tokenizer=input_count.tokenizer,
        )

    def estimate_from_counts(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        tokenizer: str = "provided-counts",
    ):
        model_pricing = self.catalog["models"][model]

        input_cost = (input_tokens / 1_000_000) * model_pricing["input_per_million"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output_per_million"]
        total_cost = input_cost + output_cost

        return CostEstimate(
            model=model,
            provider=model_pricing["provider"],
            currency=self.catalog["currency"],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=round(input_cost, 8),
            output_cost=round(output_cost, 8),
            total_cost=round(total_cost, 8),
            tokenizer=tokenizer,
        )


def calculate_savings(baseline: CostEstimate, optimized: CostEstimate):
    baseline_tokens = baseline.input_tokens + baseline.output_tokens
    optimized_tokens = optimized.input_tokens + optimized.output_tokens

    tokens_saved = baseline_tokens - optimized_tokens
    cost_saved = baseline.total_cost - optimized.total_cost

    return {
        "tokens_saved": tokens_saved,
        "token_reduction_pct": round((tokens_saved / baseline_tokens) * 100, 4),
        "cost_saved": round(cost_saved, 8),
        "cost_reduction_pct": round((cost_saved / baseline.total_cost) * 100, 4),
    }