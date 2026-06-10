from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
    ):
        self.model = CrossEncoder(model_name, device="cpu")
        self.top_n = top_n

    def rerank(self, query: str, documents: list):
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        reranked_docs = []

        for rank, (doc, score) in enumerate(ranked[: self.top_n], start=1):
            doc.metadata["rerank_score"] = float(score)
            doc.metadata["rerank_rank"] = rank
            reranked_docs.append(doc)

        return reranked_docs