import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from efficiency_gateway.cache.semantic_cache import SemanticCache


def main():
    cache = SemanticCache(
        persist_dir=str(ROOT / "data" / "chroma"),
        collection_name="semantic_response_cache",
        max_distance=0.22,
    )

    question = input("Ask a question: ")

    result = cache.lookup(question)

    if result.cache_hit:
        print("\nCACHE HIT")
        print(f"Similarity: {result.similarity}")
        print(f"Matched question: {result.matched_question}")
        print("\nCached answer:")
        print(result.answer)
        return

    print("\nCACHE MISS")
    print("No similar cached answer found.")

    answer = input("\nType an answer to store in cache for testing: ")

    cache_id = cache.store(
        question=question,
        answer=answer,
        model_name="manual_test_answer",
    )

    print("\nStored new cache entry.")
    print(f"Cache ID: {cache_id}")
    print(f"Total cache entries: {cache.count()}")


if __name__ == "__main__":
    main()