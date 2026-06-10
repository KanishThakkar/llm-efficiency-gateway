import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from efficiency_gateway.core.token_counter import TokenCounter
from efficiency_gateway.rag.context_pruner import AdaptiveContextPruner
from efficiency_gateway.rag.hybrid_retriever import HybridPaperRetriever
from efficiency_gateway.rag.prompt_builder import build_rag_prompt, format_context
from efficiency_gateway.rag.prompt_compressor import LLMLinguaPromptCompressor
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

    compressor = LLMLinguaPromptCompressor(
        rate=0.55,
    )

    token_counter = TokenCounter()

    print("\nRetrieving with hybrid search...")
    candidate_docs = retriever.retrieve(query=query, top_k=15)

    print("Reranking...")
    reranked_docs = reranker.rerank(query=query, documents=candidate_docs)

    print("Pruning context...")
    pruning_result = pruner.prune(reranked_docs)

    original_context = format_context(pruning_result.selected_documents)

    print("Compressing retrieved context...")
    compression_result = compressor.compress_text(original_context)

    final_prompt = build_rag_prompt(
        question=query,
        context=compression_result.compressed_text,
    )

    final_prompt_tokens = token_counter.count(final_prompt).tokens

    print("\nPrompt compression summary:")
    print(f"Selected chunks: {pruning_result.selected_count}")
    print(f"Original context tokens: {compression_result.original_tokens}")
    print(f"Compressed context tokens: {compression_result.compressed_tokens}")
    print(f"Tokens saved: {compression_result.tokens_saved}")
    print(f"Compression ratio: {compression_result.compression_ratio}x")
    print(f"Final prompt tokens: {final_prompt_tokens}")
    print(f"Compressor: {compression_result.compressor_name}")

    print("\nCompressed prompt preview:\n")
    print(final_prompt[:2000])


if __name__ == "__main__":
    main()