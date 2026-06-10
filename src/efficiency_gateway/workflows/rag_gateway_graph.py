import time
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from efficiency_gateway.cache.semantic_cache import SemanticCache
from efficiency_gateway.core.cost_tracker import CostTracker, calculate_savings
from efficiency_gateway.core.metrics_store import MetricsStore, RunRecord, new_run_id
from efficiency_gateway.core.token_counter import TokenCounter
from efficiency_gateway.evaluation.ragas_evaluator import RAGASEvaluator
from efficiency_gateway.llm.answer_generator import LLMAnswerGenerator
from efficiency_gateway.rag.context_pruner import AdaptiveContextPruner
from efficiency_gateway.rag.hybrid_retriever import HybridPaperRetriever
from efficiency_gateway.rag.prompt_builder import build_rag_prompt, format_context
from efficiency_gateway.rag.prompt_compressor import LLMLinguaPromptCompressor
from efficiency_gateway.rag.reranker import CrossEncoderReranker
from efficiency_gateway.routing.model_router import QualityAwareModelRouter
from efficiency_gateway.security.input_guard import InputSecurityGuard


class GatewayState(TypedDict, total=False):
    # --- input ---
    question: str
    required_quality: str
    budget_limit: float | None

    # --- security ---
    sanitized_question: str
    status: str

    # --- token counting ---
    query_tokens: int
    run_id: str
    start_time: float

    # --- cache ---
    cache_hit: bool
    cache_similarity: float | None
    cache_answer: str | None

    # --- retrieval ---
    candidates: list[Any]
    safe_docs: list[Any]
    blocked_docs: list[Any]
    reranked_docs: list[Any]
    selected_docs: list[Any]
    selected_chunks: int
    blocked_chunks: int

    # --- compression ---
    final_prompt: str | None
    original_context_tokens: int
    compressed_context_tokens: int
    compression_tokens_saved: int

    # --- routing ---
    selected_model: str
    route_reason: str
    estimated_cost: float
    budget_ok: bool
    retrieval_confidence: float
    complexity_score: float

    # --- generation ---
    answer: str | None
    answer_tokens: int
    actual_input_tokens: int
    latency_ms: float

    # --- evaluation ---
    faithfulness: float
    answer_relevancy: float
    quality_score: float
    evaluator: str

    # --- cost savings ---
    baseline_input_tokens: int
    baseline_cost: float
    tokens_saved: int
    cost_saved: float
    token_reduction_pct: float
    cost_reduction_pct: float


def build_gateway_graph(
    persist_dir: str = "data/chroma",
    collection_name: str = "research_papers",
    router_config_path: str = "configs/model_router.json",
    pricing_path: str = "configs/pricing.json",
    db_path: str = "data/metrics.db",
    ollama_api_base: str = "http://localhost:11434",
    strong_model_for_baseline: str = "groq/llama-3.3-70b-versatile",
):
    guard = InputSecurityGuard()
    token_counter = TokenCounter()
    cost_tracker = CostTracker(pricing_path=pricing_path)

    cache = SemanticCache(
        persist_dir=persist_dir,
        collection_name="semantic_response_cache",
    )
    retriever = HybridPaperRetriever(
        persist_dir=persist_dir,
        collection_name=collection_name,
    )
    reranker = CrossEncoderReranker(top_n=10)
    pruner = AdaptiveContextPruner(token_budget=1800, min_chunks=2, max_chunks=5)
    compressor = LLMLinguaPromptCompressor(rate=0.55)
    router = QualityAwareModelRouter(
        config_path=router_config_path,
        pricing_path=pricing_path,
    )
    generator = LLMAnswerGenerator(ollama_api_base=ollama_api_base)
    evaluator = RAGASEvaluator()
    metrics_store = MetricsStore(db_path=db_path)

    # ------------------------------------------------------------------ nodes

    def init_run(state: GatewayState) -> GatewayState:
        return {
            "run_id": new_run_id(),
            "start_time": time.perf_counter(),
            "query_tokens": token_counter.count(state["question"]).tokens,
        }

    def security_check(state: GatewayState) -> GatewayState:
        result = guard.scan(state["question"])
        if not result.is_valid:
            return {
                "status": "blocked",
                "sanitized_question": result.sanitized_text,
                "answer": "Request blocked: unsafe query detected.",
            }
        return {"status": "safe", "sanitized_question": result.sanitized_text}

    def cache_lookup(state: GatewayState) -> GatewayState:
        result = cache.lookup(state["sanitized_question"])
        if result.cache_hit:
            return {
                "status": "cache_hit",
                "cache_hit": True,
                "cache_similarity": result.similarity,
                "cache_answer": result.answer,
                "answer": result.answer,
                "selected_model": "cache",
                "estimated_cost": 0.0,
                "tokens_saved": state.get("query_tokens", 0),
                "cost_saved": 0.0,
                "token_reduction_pct": 100.0,
                "cost_reduction_pct": 100.0,
                "quality_score": 1.0,
                "faithfulness": 1.0,
                "answer_relevancy": 1.0,
                "evaluator": "cache",
            }
        return {"status": "cache_miss", "cache_hit": False, "cache_similarity": result.similarity}

    def retrieve(state: GatewayState) -> GatewayState:
        candidates = retriever.retrieve(query=state["sanitized_question"], top_k=15)
        return {"candidates": candidates}

    def scan_retrieved_context(state: GatewayState) -> GatewayState:
        safe_docs, blocked_docs = guard.scan_documents(state["candidates"])
        if not safe_docs:
            return {
                "status": "no_safe_context",
                "safe_docs": [],
                "blocked_docs": blocked_docs,
                "blocked_chunks": len(blocked_docs),
                "answer": "No safe retrieved context available.",
            }
        return {
            "status": "context_safe",
            "safe_docs": safe_docs,
            "blocked_docs": blocked_docs,
            "blocked_chunks": len(blocked_docs),
        }

    def rerank(state: GatewayState) -> GatewayState:
        docs = reranker.rerank(query=state["sanitized_question"], documents=state["safe_docs"])
        return {"reranked_docs": docs}

    def prune_context(state: GatewayState) -> GatewayState:
        result = pruner.prune(state["reranked_docs"])
        return {
            "selected_docs": result.selected_documents,
            "selected_chunks": result.selected_count,
        }

    def compress_prompt(state: GatewayState) -> GatewayState:
        context = format_context(state["selected_docs"])
        compression = compressor.compress_text(context)
        final_prompt = build_rag_prompt(
            question=state["sanitized_question"],
            context=compression.compressed_text,
        )
        return {
            "final_prompt": final_prompt,
            "original_context_tokens": compression.original_tokens,
            "compressed_context_tokens": compression.compressed_tokens,
            "compression_tokens_saved": compression.tokens_saved,
        }

    def route_model(state: GatewayState) -> GatewayState:
        decision = router.route(
            question=state["sanitized_question"],
            final_prompt=state["final_prompt"],
            selected_documents=state["selected_docs"],
            required_quality=state.get("required_quality", "normal"),
            budget_limit=state.get("budget_limit"),
            security_risk=0.0,
            estimated_output_tokens=600,
        )
        return {
            "status": "ready_for_model_call",
            "selected_model": decision.selected_model,
            "route_reason": decision.route_reason,
            "estimated_cost": decision.estimated_cost,
            "budget_ok": decision.budget_ok,
            "retrieval_confidence": decision.retrieval_confidence,
            "complexity_score": decision.complexity_score,
        }

    def generate_answer(state: GatewayState) -> GatewayState:
        result = generator.generate(
            model=state["selected_model"],
            prompt=state["final_prompt"],
        )
        return {
            "status": "answer_generated",
            "answer": result.answer,
            "answer_tokens": result.output_tokens,
            "actual_input_tokens": result.input_tokens,
            "latency_ms": round((time.perf_counter() - state["start_time"]) * 1000, 2),
        }

    def store_to_cache(state: GatewayState) -> GatewayState:
        cache.store(
            question=state["sanitized_question"],
            answer=state["answer"],
            model_name=state["selected_model"],
            estimated_cost=state.get("estimated_cost", 0.0),
            input_tokens=state.get("actual_input_tokens", 0),
            output_tokens=state.get("answer_tokens", 0),
        )
        return {}

    def evaluate_quality(state: GatewayState) -> GatewayState:
        contexts = [doc.page_content for doc in state.get("selected_docs", [])]
        eval_result = evaluator.evaluate(
            question=state["sanitized_question"],
            answer=state["answer"],
            contexts=contexts,
        )
        return {
            "faithfulness": eval_result.faithfulness,
            "answer_relevancy": eval_result.answer_relevancy,
            "quality_score": eval_result.quality_score,
            "evaluator": eval_result.evaluator,
        }

    def compute_savings(state: GatewayState) -> GatewayState:
        """Compute token/cost savings vs a no-optimisation baseline (full context → strong model)."""
        original_ctx = state.get("original_context_tokens", 0)
        query_tok = state.get("query_tokens", 0)
        answer_tok = state.get("answer_tokens", 0)

        baseline_input = original_ctx + query_tok
        optimized_input = state.get("actual_input_tokens", state.get("compressed_context_tokens", 0))

        baseline_est = cost_tracker.estimate_from_counts(
            model=strong_model_for_baseline,
            input_tokens=baseline_input,
            output_tokens=answer_tok,
        )
        optimized_est = cost_tracker.estimate_from_counts(
            model=state.get("selected_model", strong_model_for_baseline),
            input_tokens=optimized_input,
            output_tokens=answer_tok,
        )

        savings = calculate_savings(baseline_est, optimized_est)

        return {
            "baseline_input_tokens": baseline_input,
            "baseline_cost": baseline_est.total_cost,
            "tokens_saved": savings["tokens_saved"],
            "cost_saved": savings["cost_saved"],
            "token_reduction_pct": savings["token_reduction_pct"],
            "cost_reduction_pct": savings["cost_reduction_pct"],
        }

    def track_metrics(state: GatewayState) -> GatewayState:
        elapsed = round((time.perf_counter() - state["start_time"]) * 1000, 2)

        record = RunRecord(
            run_id=state.get("run_id", new_run_id()),
            question=state.get("question", ""),
            status=state.get("status", "unknown"),
            cache_hit=int(state.get("cache_hit", False)),
            selected_model=state.get("selected_model", ""),
            input_tokens=state.get("actual_input_tokens", 0),
            output_tokens=state.get("answer_tokens", 0),
            baseline_input_tokens=state.get("baseline_input_tokens", 0),
            estimated_cost=state.get("estimated_cost", 0.0),
            baseline_cost=state.get("baseline_cost", 0.0),
            tokens_saved=state.get("tokens_saved", 0),
            cost_saved=state.get("cost_saved", 0.0),
            token_reduction_pct=state.get("token_reduction_pct", 0.0),
            cost_reduction_pct=state.get("cost_reduction_pct", 0.0),
            latency_ms=state.get("latency_ms", elapsed),
            quality_score=state.get("quality_score", 0.0),
            faithfulness=state.get("faithfulness", 0.0),
            answer_relevancy=state.get("answer_relevancy", 0.0),
            evaluator=state.get("evaluator", ""),
            timestamp=time.time(),
        )
        metrics_store.save(record)
        return {}

    # ------------------------------------------------------------------ edges

    def after_security(state: GatewayState) -> Literal["cache_lookup", "__end__"]:
        return END if state["status"] == "blocked" else "cache_lookup"

    def after_cache(state: GatewayState) -> Literal["retrieve", "track_metrics"]:
        return "track_metrics" if state["status"] == "cache_hit" else "retrieve"

    def after_context_scan(state: GatewayState) -> Literal["rerank", "__end__"]:
        return END if state["status"] == "no_safe_context" else "rerank"

    # ------------------------------------------------------------------ graph

    graph = StateGraph(GatewayState)

    graph.add_node("init_run", init_run)
    graph.add_node("security_check", security_check)
    graph.add_node("cache_lookup", cache_lookup)
    graph.add_node("retrieve", retrieve)
    graph.add_node("scan_retrieved_context", scan_retrieved_context)
    graph.add_node("rerank", rerank)
    graph.add_node("prune_context", prune_context)
    graph.add_node("compress_prompt", compress_prompt)
    graph.add_node("route_model", route_model)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("store_to_cache", store_to_cache)
    graph.add_node("evaluate_quality", evaluate_quality)
    graph.add_node("compute_savings", compute_savings)
    graph.add_node("track_metrics", track_metrics)

    graph.add_edge(START, "init_run")
    graph.add_edge("init_run", "security_check")
    graph.add_conditional_edges("security_check", after_security)
    graph.add_conditional_edges("cache_lookup", after_cache)

    # cache-miss path
    graph.add_edge("retrieve", "scan_retrieved_context")
    graph.add_conditional_edges("scan_retrieved_context", after_context_scan)
    graph.add_edge("rerank", "prune_context")
    graph.add_edge("prune_context", "compress_prompt")
    graph.add_edge("compress_prompt", "route_model")
    graph.add_edge("route_model", "generate_answer")
    graph.add_edge("generate_answer", "store_to_cache")
    graph.add_edge("store_to_cache", "evaluate_quality")
    graph.add_edge("evaluate_quality", "compute_savings")
    graph.add_edge("compute_savings", "track_metrics")
    graph.add_edge("track_metrics", END)

    return graph.compile()
