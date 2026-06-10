import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from efficiency_gateway.rag.context_pruner import AdaptiveContextPruner
from efficiency_gateway.rag.hybrid_retriever import HybridPaperRetriever
from efficiency_gateway.rag.prompt_builder import build_rag_prompt, format_context
from efficiency_gateway.rag.prompt_compressor import LLMLinguaPromptCompressor
from efficiency_gateway.rag.reranker import CrossEncoderReranker
from efficiency_gateway.routing.model_router import QualityAwareModelRouter
from efficiency_gateway.security.input_guard import InputSecurityGuard


def main():
    query = input("Ask a question: ")

    quality = input("Required quality? low / normal / high: ").strip().lower() or "normal"
    budget_text = input("Budget limit in USD? Press Enter for no budget: ").strip()

    budget_limit = float(budget_text) if budget_text else None

    guard = InputSecurityGuard()
    query_guard = guard.scan(query)

    if not query_guard.is_valid:
        print("\nBlocked unsafe query.")
        print(f"Risk score: {query_guard.max_risk_score}")
        print(f"Sanitized query: {query_guard.sanitized_text}")
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

    router = QualityAwareModelRouter(
        config_path=str(ROOT / "configs" / "model_router.json"),
        pricing_path=str(ROOT / "configs" / "pricing.json"),
    )

    print("\nRetrieving...")
    candidates = retriever.retrieve(query=query_guard.sanitized_text, top_k=15)

    print("Scanning retrieved chunks...")
    safe_docs, blocked_docs = guard.scan_documents(candidates)

    print("Reranking...")
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

    decision = router.route(
        question=query_guard.sanitized_text,
        final_prompt=final_prompt,
        selected_documents=pruning_result.selected_documents,
        required_quality=quality,
        budget_limit=budget_limit,
        security_risk=query_guard.max_risk_score,
        estimated_output_tokens=600,
    )

    print("\nModel routing decision:")
    print(f"Selected model: {decision.selected_model}")
    print(f"Reason: {decision.route_reason}")
    print(f"Query tokens: {decision.query_tokens}")
    print(f"Prompt tokens: {decision.prompt_tokens}")
    print(f"Context tokens: {decision.context_tokens}")
    print(f"Estimated output tokens: {decision.estimated_output_tokens}")
    print(f"Retrieval confidence: {decision.retrieval_confidence}")
    print(f"Complexity score: {decision.complexity_score}")
    print(f"Estimated cost: ${decision.estimated_cost}")
    print(f"Budget limit: {decision.budget_limit}")
    print(f"Budget OK: {decision.budget_ok}")
    print(f"Blocked chunks: {len(blocked_docs)}")


if __name__ == "__main__":
    main()