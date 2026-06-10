import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from efficiency_gateway.rag.context_pruner import AdaptiveContextPruner
from efficiency_gateway.rag.hybrid_retriever import HybridPaperRetriever
from efficiency_gateway.rag.prompt_builder import build_rag_prompt, format_context
from efficiency_gateway.rag.prompt_compressor import LLMLinguaPromptCompressor
from efficiency_gateway.rag.reranker import CrossEncoderReranker
from efficiency_gateway.security.input_guard import InputSecurityGuard


def main():
    query = input("Ask a question: ")

    guard = InputSecurityGuard()

    query_guard = guard.scan(query)

    if not query_guard.is_valid:
        print("\nBlocked unsafe user query.")
        print(f"Risk score: {query_guard.max_risk_score}")
        print(f"Sanitized version: {query_guard.sanitized_text}")
        return

    retriever = HybridPaperRetriever(
        persist_dir=str(ROOT / "data" / "chroma"),
        collection_name="research_papers",
    )

    reranker = CrossEncoderReranker(top_n=10)

    pruner = AdaptiveContextPruner(
        token_budget=1800,
        min_chunks=2,
        max_chunks=5,
    )

    compressor = LLMLinguaPromptCompressor(rate=0.55)

    print("\nRetrieving...")
    candidate_docs = retriever.retrieve(query=query_guard.sanitized_text, top_k=15)

    print("Scanning retrieved chunks for injection/PII/secrets...")
    safe_docs, blocked_docs = guard.scan_documents(candidate_docs)

    print(f"Safe chunks: {len(safe_docs)}")
    print(f"Blocked chunks: {len(blocked_docs)}")

    if not safe_docs:
        print("No safe retrieved context available.")
        return

    print("Reranking safe chunks...")
    reranked_docs = reranker.rerank(query=query_guard.sanitized_text, documents=safe_docs)

    print("Pruning context...")
    pruning_result = pruner.prune(reranked_docs)

    context = format_context(pruning_result.selected_documents)

    print("Compressing context...")
    compression_result = compressor.compress_text(context)

    final_prompt = build_rag_prompt(
        question=query_guard.sanitized_text,
        context=compression_result.compressed_text,
    )

    print("\nSecure RAG prompt ready.")
    print(f"Selected chunks: {pruning_result.selected_count}")
    print(f"Blocked chunks: {len(blocked_docs)}")
    print(f"Original context tokens: {compression_result.original_tokens}")
    print(f"Compressed context tokens: {compression_result.compressed_tokens}")
    print(f"Tokens saved: {compression_result.tokens_saved}")

    print("\nPrompt preview:\n")
    print(final_prompt[:2000])


if __name__ == "__main__":
    main()