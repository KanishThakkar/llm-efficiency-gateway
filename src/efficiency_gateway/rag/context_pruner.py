from dataclasses import dataclass

from efficiency_gateway.core.token_counter import TokenCounter


@dataclass(frozen=True)
class PruningResult:
    selected_documents: list
    rejected_documents: list
    total_context_tokens: int
    token_budget: int
    selected_count: int
    rejected_count: int


class AdaptiveContextPruner:
    def __init__(
        self,
        token_budget: int = 1800,
        min_chunks: int = 2,
        max_chunks: int = 6,
        min_rerank_score: float | None = None,
    ):
        self.token_budget = token_budget
        self.min_chunks = min_chunks
        self.max_chunks = max_chunks
        self.min_rerank_score = min_rerank_score
        self.token_counter = TokenCounter()

    def prune(self, documents: list) -> PruningResult:
        selected = []
        rejected = []
        total_tokens = 0

        sorted_docs = sorted(
            documents,
            key=lambda doc: doc.metadata.get("rerank_score", 0),
            reverse=True,
        )

        for doc in sorted_docs:
            if len(selected) >= self.max_chunks:
                rejected.append(doc)
                continue

            score = doc.metadata.get("rerank_score")

            if (
                self.min_rerank_score is not None
                and score is not None
                and score < self.min_rerank_score
                and len(selected) >= self.min_chunks
            ):
                rejected.append(doc)
                continue

            chunk_tokens = doc.metadata.get("token_count")

            if chunk_tokens is None:
                chunk_tokens = self.token_counter.count(doc.page_content).tokens
                doc.metadata["token_count"] = chunk_tokens

            would_exceed_budget = total_tokens + chunk_tokens > self.token_budget

            if would_exceed_budget and len(selected) >= self.min_chunks:
                rejected.append(doc)
                continue

            selected.append(doc)
            total_tokens += chunk_tokens

        for rank, doc in enumerate(selected, start=1):
            doc.metadata["pruned_rank"] = rank
            doc.metadata["included_in_context"] = True

        for doc in rejected:
            doc.metadata["included_in_context"] = False

        return PruningResult(
            selected_documents=selected,
            rejected_documents=rejected,
            total_context_tokens=total_tokens,
            token_budget=self.token_budget,
            selected_count=len(selected),
            rejected_count=len(rejected),
        )