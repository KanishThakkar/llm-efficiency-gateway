from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


@dataclass(frozen=True)
class CacheLookupResult:
    cache_hit: bool
    answer: str | None
    matched_question: str | None
    distance: float | None
    similarity: float | None
    metadata: dict | None


class SemanticCache:
    def __init__(
        self,
        persist_dir: str = "data/chroma",
        collection_name: str = "semantic_response_cache",
        embedding_model: str = "all-MiniLM-L6-v2",
        max_distance: float = 0.22,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.max_distance = max_distance

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def lookup(
        self,
        question: str,
        corpus_version: str = "research_papers_v1",
        top_k: int = 1,
    ) -> CacheLookupResult:
        if self.collection.count() == 0:
            return CacheLookupResult(
                cache_hit=False,
                answer=None,
                matched_question=None,
                distance=None,
                similarity=None,
                metadata=None,
            )

        results = self.collection.query(
            query_texts=[question],
            n_results=top_k,
            where={"corpus_version": corpus_version},
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not documents or not metadatas or not distances:
            return CacheLookupResult(
                cache_hit=False,
                answer=None,
                matched_question=None,
                distance=None,
                similarity=None,
                metadata=None,
            )

        distance = float(distances[0])
        metadata = metadatas[0]
        matched_question = documents[0]

        if distance > self.max_distance:
            return CacheLookupResult(
                cache_hit=False,
                answer=None,
                matched_question=matched_question,
                distance=distance,
                similarity=round(1 - distance, 4),
                metadata=metadata,
            )

        return CacheLookupResult(
            cache_hit=True,
            answer=metadata.get("answer"),
            matched_question=matched_question,
            distance=distance,
            similarity=round(1 - distance, 4),
            metadata=metadata,
        )

    def store(
        self,
        question: str,
        answer: str,
        corpus_version: str = "research_papers_v1",
        model_name: str = "not_called_yet",
        estimated_cost: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> str:
        cache_id = str(uuid4())

        metadata = {
            "answer": answer,
            "corpus_version": corpus_version,
            "model_name": model_name,
            "estimated_cost": float(estimated_cost),
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        self.collection.upsert(
            ids=[cache_id],
            documents=[question],
            metadatas=[metadata],
        )

        return cache_id

    def count(self) -> int:
        return self.collection.count()