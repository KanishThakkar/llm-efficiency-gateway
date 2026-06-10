import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from efficiency_gateway.rag.vector_store import PaperVectorStore


def main():
    vector_store = PaperVectorStore(
        persist_dir=str(ROOT / "data" / "chroma"),
        collection_name="research_papers",
    )

    query = input("Ask a question: ")

    results = vector_store.query(query, top_k=5)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (doc, meta, distance) in enumerate(zip(docs, metas, distances), start=1):
        print("=" * 80)
        print(f"Rank: {rank}")
        print(f"Paper: {meta.get('paper_title')}")
        print(f"Page: {meta.get('page_number')}")
        print(f"Token count: {meta.get('token_count')}")
        print(f"Distance: {distance}")
        print("-" * 80)
        print(doc[:900])
        print()


if __name__ == "__main__":
    main()