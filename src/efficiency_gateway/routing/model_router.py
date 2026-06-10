import json
import math
from dataclasses import dataclass
from pathlib import Path

from efficiency_gateway.core.cost_tracker import CostTracker
from efficiency_gateway.core.token_counter import TokenCounter


@dataclass(frozen=True)
class RoutingDecision:
    selected_model: str
    route_reason: str
    query_tokens: int
    prompt_tokens: int
    context_tokens: int
    estimated_output_tokens: int
    retrieval_confidence: float
    complexity_score: float
    estimated_cost: float
    budget_limit: float | None
    budget_ok: bool


class QualityAwareModelRouter:
    def __init__(
        self,
        config_path: str = "configs/model_router.json",
        pricing_path: str = "configs/pricing.json",
    ):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.token_counter = TokenCounter()
        self.cost_tracker = CostTracker(pricing_path=pricing_path)

        defaults = self.config["defaults"]
        self.local_model = defaults["local_model"]
        self.cheap_model = defaults["cheap_model"]
        self.strong_model = defaults["strong_model"]
        self.default_output_tokens = int(defaults["default_output_tokens"])

    def _load_config(self):
        with self.config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def route(
        self,
        question: str,
        final_prompt: str,
        selected_documents: list,
        required_quality: str = "normal",
        budget_limit: float | None = None,
        security_risk: float = 0.0,
        estimated_output_tokens: int | None = None,
    ) -> RoutingDecision:
        query_tokens = self.token_counter.count(question).tokens
        prompt_tokens = self.token_counter.count(final_prompt).tokens
        context_tokens = self._context_tokens(selected_documents)

        output_tokens = estimated_output_tokens or self.default_output_tokens
        retrieval_confidence = self._retrieval_confidence(selected_documents)

        complexity_score = self._complexity_score(
            question=question,
            query_tokens=query_tokens,
            prompt_tokens=prompt_tokens,
            context_tokens=context_tokens,
            retrieval_confidence=retrieval_confidence,
            required_quality=required_quality,
            security_risk=security_risk,
        )

        selected_model, reason = self._select_model(
            complexity_score=complexity_score,
            retrieval_confidence=retrieval_confidence,
            required_quality=required_quality,
            prompt_tokens=prompt_tokens,
            security_risk=security_risk,
        )

        estimate = self.cost_tracker.estimate_from_counts(
            model=selected_model,
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )

        budget_ok = True

        if budget_limit is not None and estimate.total_cost > budget_limit:
            selected_model, reason = self._downgrade_for_budget(
                selected_model=selected_model,
                current_reason=reason,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                budget_limit=budget_limit,
            )

            estimate = self.cost_tracker.estimate_from_counts(
                model=selected_model,
                input_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )

            budget_ok = estimate.total_cost <= budget_limit

        return RoutingDecision(
            selected_model=selected_model,
            route_reason=reason,
            query_tokens=query_tokens,
            prompt_tokens=prompt_tokens,
            context_tokens=context_tokens,
            estimated_output_tokens=output_tokens,
            retrieval_confidence=round(retrieval_confidence, 4),
            complexity_score=round(complexity_score, 4),
            estimated_cost=estimate.total_cost,
            budget_limit=budget_limit,
            budget_ok=budget_ok,
        )

    def _context_tokens(self, documents: list) -> int:
        total = 0

        for doc in documents:
            token_count = doc.metadata.get("token_count")

            if token_count is None:
                token_count = self.token_counter.count(doc.page_content).tokens

            total += int(token_count)

        return total

    def _retrieval_confidence(self, documents: list) -> float:
        scores = []

        for doc in documents:
            score = doc.metadata.get("rerank_score")

            if score is not None:
                scores.append(float(score))

        if not scores:
            return 0.5

        probabilities = [1 / (1 + math.exp(-score)) for score in scores]
        top_scores = sorted(probabilities, reverse=True)[:3]

        return sum(top_scores) / len(top_scores)

    def _complexity_score(
        self,
        question: str,
        query_tokens: int,
        prompt_tokens: int,
        context_tokens: int,
        retrieval_confidence: float,
        required_quality: str,
        security_risk: float,
    ) -> float:
        score = 0.0

        if query_tokens > 35:
            score += 0.15

        if prompt_tokens > 1800:
            score += 0.20

        if context_tokens > 1400:
            score += 0.15

        if retrieval_confidence < 0.55:
            score += 0.25

        if required_quality == "high":
            score += 0.25

        if security_risk > 0.5:
            score += 0.15

        hard_words = [
            "compare",
            "prove",
            "derive",
            "analyze",
            "why",
            "tradeoff",
            "limitations",
            "failure",
            "mathematical",
            "architecture",
        ]

        lower_question = question.lower()

        if any(word in lower_question for word in hard_words):
            score += 0.15

        return min(score, 1.0)

    def _select_model(
        self,
        complexity_score: float,
        retrieval_confidence: float,
        required_quality: str,
        prompt_tokens: int,
        security_risk: float,
    ) -> tuple[str, str]:
        if required_quality == "high":
            return self.strong_model, "High-quality answer requested."

        if security_risk > 0.7:
            return self.strong_model, "Higher-risk request needs stronger reasoning."

        if retrieval_confidence < 0.45:
            return self.strong_model, "Low retrieval confidence."

        if complexity_score >= 0.55:
            return self.strong_model, "High complexity score."

        if complexity_score <= 0.15 and prompt_tokens < 900 and retrieval_confidence > 0.70:
            return self.local_model, "Simple, short, high-confidence request."

        return self.cheap_model, "Normal request routed to cheap model."

    def _downgrade_for_budget(
        self,
        selected_model: str,
        current_reason: str,
        prompt_tokens: int,
        output_tokens: int,
        budget_limit: float,
    ) -> tuple[str, str]:
        candidate_order = [self.local_model, self.cheap_model, self.strong_model]

        if selected_model == self.strong_model:
            candidate_order = [self.cheap_model, self.local_model]
        elif selected_model == self.cheap_model:
            candidate_order = [self.local_model]

        for candidate in candidate_order:
            estimate = self.cost_tracker.estimate_from_counts(
                model=candidate,
                input_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )

            if estimate.total_cost <= budget_limit:
                return (
                    candidate,
                    f"{current_reason} Downgraded to fit budget limit.",
                )

        return (
            self.local_model,
            f"{current_reason} Budget too low for paid models.",
        )