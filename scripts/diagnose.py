"""
Diagnose script — tests each pipeline component one at a time.
Run this first to find exactly which step is failing.
    python scripts/diagnose.py
"""
import logging
import os
import sys
import traceback
from pathlib import Path

logging.basicConfig(level=logging.ERROR)
for _lib in ("llm_guard", "transformers", "torch", "sentence_transformers",
             "chromadb", "httpx", "huggingface_hub", "llmlingua"):
    logging.getLogger(_lib).setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PASS = "  [PASS]"
FAIL = "  [FAIL]"


def check(label: str, fn):
    print(f"\n{label} ...", end=" ")
    sys.stdout.flush()
    try:
        result = fn()
        print(f"OK  {result or ''}")
        return True
    except BaseException:
        print("FAILED")
        traceback.print_exc()
        return False


def main():
    print("=" * 55)
    print("  LLM Efficiency Gateway — Diagnostics")
    print("=" * 55)

    # 1. token counter
    ok = check("1. TokenCounter", lambda: (
        __import__("efficiency_gateway.core.token_counter", fromlist=["TokenCounter"])
        .TokenCounter().count("hello world").tokens
        and "tokens ok"
    ))
    if not ok:
        return

    # 2. cost tracker
    ok = check("2. CostTracker", lambda: (
        __import__("efficiency_gateway.core.cost_tracker", fromlist=["CostTracker"])
        .CostTracker(pricing_path=str(ROOT / "configs" / "pricing.json"))
        .estimate_from_counts("groq/llama-3.1-8b-instant", 100, 50)
        and "pricing ok"
    ))

    # 3. chromadb
    def _chroma():
        import chromadb
        client = chromadb.PersistentClient(path=str(ROOT / "data" / "chroma"))
        col = client.get_collection("research_papers")
        n = col.count()
        if n == 0:
            raise ValueError("ChromaDB is empty — run ingest_papers_step2.py first")
        return f"{n} chunks"
    ok = check("3. ChromaDB (research_papers)", _chroma)
    if not ok:
        return

    # 4. semantic cache
    check("4. SemanticCache", lambda: (
        __import__("efficiency_gateway.cache.semantic_cache", fromlist=["SemanticCache"])
        .SemanticCache(persist_dir=str(ROOT / "data" / "chroma"))
        and "cache ok"
    ))

    # 5. security guard
    def _guard():
        from efficiency_gateway.security.input_guard import InputSecurityGuard
        g = InputSecurityGuard()
        r = g.scan("What is attention in transformers?")
        return f"valid={r.is_valid} risk={r.max_risk_score:.2f}"
    check("5. InputSecurityGuard", _guard)

    # 6. hybrid retriever
    def _retriever():
        from efficiency_gateway.rag.hybrid_retriever import HybridPaperRetriever
        r = HybridPaperRetriever(
            persist_dir=str(ROOT / "data" / "chroma"),
            collection_name="research_papers",
        )
        docs = r.retrieve("attention mechanism", top_k=3)
        return f"{len(docs)} docs retrieved"
    ok = check("6. HybridRetriever (BM25 + vector)", _retriever)
    if not ok:
        return

    # 7. reranker
    def _reranker():
        from langchain_core.documents import Document
        from efficiency_gateway.rag.reranker import CrossEncoderReranker
        rr = CrossEncoderReranker(top_n=3)
        docs = [Document(page_content="attention is used in transformers", metadata={})]
        out = rr.rerank("attention mechanism", docs)
        return f"{len(out)} docs reranked"
    check("7. CrossEncoderReranker", _reranker)

    # 8. prompt compressor
    def _compressor():
        from efficiency_gateway.rag.prompt_compressor import LLMLinguaPromptCompressor
        c = LLMLinguaPromptCompressor(rate=0.55)
        r = c.compress_text("The attention mechanism is a key component of transformer models. " * 10)
        return f"compressor={r.compressor_name} tokens={r.original_tokens}->{r.compressed_tokens}"
    check("8. PromptCompressor (LLMLingua / fallback)", _compressor)

    # 9. model router
    def _router():
        from langchain_core.documents import Document
        from efficiency_gateway.routing.model_router import QualityAwareModelRouter
        rtr = QualityAwareModelRouter(
            config_path=str(ROOT / "configs" / "model_router.json"),
            pricing_path=str(ROOT / "configs" / "pricing.json"),
        )
        doc = Document(page_content="test", metadata={"rerank_score": 0.8})
        d = rtr.route("What is attention?", "prompt text", [doc])
        return f"model={d.selected_model} cost=${d.estimated_cost:.6f}"
    check("9. ModelRouter", _router)

    # 10. groq api key + litellm
    def _llm():
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Run: $env:GROQ_API_KEY = 'your-key'"
            )
        from litellm import completion
        resp = completion(
            model="groq/llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Say HELLO in one word."}],
            max_tokens=5,
        )
        return f"answer='{resp.choices[0].message.content.strip()}'"
    ok = check("10. Groq LLM call (litellm)", _llm)
    if not ok:
        return

    # 11. ragas evaluator
    def _eval():
        from efficiency_gateway.evaluation.ragas_evaluator import RAGASEvaluator
        ev = RAGASEvaluator()
        r = ev.evaluate(
            question="What is attention?",
            answer="Attention is a mechanism in transformers.",
            contexts=["Transformers use an attention mechanism to weigh tokens."],
        )
        return f"quality={r.quality_score:.3f} via={r.evaluator}"
    check("11. RAGASEvaluator", _eval)

    # 12. metrics store
    check("12. MetricsStore (SQLite)", lambda: (
        __import__("efficiency_gateway.core.metrics_store", fromlist=["MetricsStore"])
        .MetricsStore(db_path=str(ROOT / "data" / "metrics.db"))
        and "db ok"
    ))

    print("\n" + "=" * 55)
    print("  All checks done. Fix any [FAIL] items above, then")
    print("  run:  python scripts/run_workflow_step10.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
