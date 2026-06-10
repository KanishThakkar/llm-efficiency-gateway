import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from efficiency_gateway.rag.hybrid_retriever import HybridPaperRetriever
from efficiency_gateway.rag.reranker import CrossEncoderReranker


def main():
    query = input("Ask a question: ")

    retriever = HybridPaperRetriever(
        persist_dir=str(ROOT / "data" / "chroma"),
        collection_name="research_papers",
    )

    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=5,
    )

    print("\nRetrieving candidates with hybrid search...")
    candidate_docs = retriever.retrieve(query=query, top_k=12)

    print(f"Hybrid candidates retrieved: {len(candidate_docs)}")

    print("Reranking candidates...")
    reranked_docs = reranker.rerank(query=query, documents=candidate_docs)

    print("\nReranked results:\n")

    for doc in reranked_docs:
        meta = doc.metadata

        print("=" * 80)
        print(f"Rerank Rank: {meta.get('rerank_rank')}")
        print(f"Rerank Score: {meta.get('rerank_score')}")
        print(f"Paper: {meta.get('paper_title')}")
        print(f"Page: {meta.get('page_number')}")
        print(f"Chunk ID: {meta.get('chunk_id')}")
        print(f"Token count: {meta.get('token_count')}")
        print("-" * 80)
        print(doc.page_content[:900])
        print()


if __name__ == "__main__":
    main()