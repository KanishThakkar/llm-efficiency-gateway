import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from efficiency_gateway.rag.hybrid_retriever import HybridPaperRetriever


def main():
    retriever = HybridPaperRetriever(
        persist_dir=str(ROOT / "data" / "chroma"),
        collection_name="research_papers",
    )

    query = input("Ask a question: ")

    results = retriever.retrieve(query=query, top_k=5)

    print("\nHybrid search results:\n")

    for rank, doc in enumerate(results, start=1):
        meta = doc.metadata

        print("=" * 80)
        print(f"Rank: {rank}")
        print(f"Paper: {meta.get('paper_title')}")
        print(f"Page: {meta.get('page_number')}")
        print(f"Chunk ID: {meta.get('chunk_id')}")
        print(f"Token count: {meta.get('token_count')}")
        print("-" * 80)
        print(doc.page_content[:900])
        print()


if __name__ == "__main__":
    main()