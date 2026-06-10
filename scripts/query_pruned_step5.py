import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from efficiency_gateway.rag.context_pruner import AdaptiveContextPruner
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
        top_n=10,
    )

    pruner = AdaptiveContextPruner(
        token_budget=1800,
        min_chunks=2,
        max_chunks=5,
    )

    print("\nRetrieving with hybrid search...")
    candidate_docs = retriever.retrieve(query=query, top_k=15)

    print(f"Hybrid candidates: {len(candidate_docs)}")

    print("Reranking...")
    reranked_docs = reranker.rerank(query=query, documents=candidate_docs)

    print("Pruning context...")
    result = pruner.prune(reranked_docs)

    print("\nPruning summary:")
    print(f"Selected chunks: {result.selected_count}")
    print(f"Rejected chunks: {result.rejected_count}")
    print(f"Context tokens: {result.total_context_tokens}/{result.token_budget}")

    print("\nSelected context chunks:\n")

    for doc in result.selected_documents:
        meta = doc.metadata

        print("=" * 80)
        print(f"Pruned Rank: {meta.get('pruned_rank')}")
        print(f"Rerank Score: {meta.get('rerank_score')}")
        print(f"Paper: {meta.get('paper_title')}")
        print(f"Page: {meta.get('page_number')}")
        print(f"Chunk ID: {meta.get('chunk_id')}")
        print(f"Token count: {meta.get('token_count')}")
        print("-" * 80)
        print(doc.page_content[:700])
        print()


if __name__ == "__main__":
    main()