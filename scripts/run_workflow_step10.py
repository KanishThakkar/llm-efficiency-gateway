"""
Full end-to-end gateway run.
All output is printed to the terminal AND saved to data/last_run.log
so nothing gets lost due to terminal scrolling.
"""
import logging
import os
import sys
import traceback
from pathlib import Path

# --- suppress noisy library logs before any imports ---
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.basicConfig(level=logging.ERROR)
for _lib in (
    "llm_guard", "transformers", "torch", "sentence_transformers",
    "chromadb", "httpx", "httpcore", "huggingface_hub", "llmlingua",
):
    logging.getLogger(_lib).setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LOG_FILE = ROOT / "data" / "last_run.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


class Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, filepath: Path):
        self._file = filepath.open("w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._stdout.flush()
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


def _fmt(val, decimals: int = 4) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def main():
    tee = Tee(LOG_FILE)
    sys.stdout = tee

    try:
        _run()
    except BaseException:
        print("\n" + "=" * 60)
        print("[FATAL ERROR] The pipeline crashed:")
        print("=" * 60)
        traceback.print_exc()
    finally:
        sys.stdout = tee._stdout
        tee.close()
        print(f"\n[Log saved to: {LOG_FILE}]")


def _run():
    # collect inputs BEFORE heavy imports so the prompt is instant
    question = input("Ask a question: ").strip()
    if not question:
        print("No question entered. Exiting.")
        return
    quality = (
        input("Required quality? [low / normal / high] (default: normal): ")
        .strip().lower() or "normal"
    )
    budget_text = input("Budget limit in USD? (press Enter for none): ").strip()
    budget_limit = float(budget_text) if budget_text else None

    # --- check API key now so we get a clear error ---
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        print(
            "\n[ERROR] GROQ_API_KEY is not set.\n"
            "Run:  $env:GROQ_API_KEY = 'your-key-here'\n"
            "Then try again."
        )
        return

    print("\nLoading pipeline components (first run may take ~30s)...")
    sys.stdout.flush()

    from efficiency_gateway.workflows.rag_gateway_graph import build_gateway_graph

    print("Building graph...")
    sys.stdout.flush()

    app = build_gateway_graph(
        persist_dir=str(ROOT / "data" / "chroma"),
        collection_name="research_papers",
        router_config_path=str(ROOT / "configs" / "model_router.json"),
        pricing_path=str(ROOT / "configs" / "pricing.json"),
        db_path=str(ROOT / "data" / "metrics.db"),
    )

    print("Running pipeline...\n")
    sys.stdout.flush()

    result = app.invoke(
        {
            "question": question,
            "required_quality": quality,
            "budget_limit": budget_limit,
        }
    )

    print("\n" + "=" * 60)
    print("GATEWAY RESULT")
    print("=" * 60)

    print(f"\n[Pipeline]")
    print(f"  Run ID          : {result.get('run_id')}")
    print(f"  Status          : {result.get('status')}")
    print(f"  Cache hit       : {result.get('cache_hit')}")
    print(f"  Selected model  : {result.get('selected_model')}")
    print(f"  Route reason    : {result.get('route_reason')}")

    print(f"\n[Retrieval]")
    print(f"  Retrieval conf  : {_fmt(result.get('retrieval_confidence'))}")
    print(f"  Complexity score: {_fmt(result.get('complexity_score'))}")
    print(f"  Selected chunks : {result.get('selected_chunks')}")
    print(f"  Blocked chunks  : {result.get('blocked_chunks')}")

    print(f"\n[Tokens]")
    print(f"  Query tokens    : {result.get('query_tokens')}")
    print(f"  Baseline input  : {result.get('baseline_input_tokens')}")
    print(f"  Optimised input : {result.get('actual_input_tokens')}")
    print(f"  Orig ctx tokens : {result.get('original_context_tokens')}")
    print(f"  Compr ctx tokens: {result.get('compressed_context_tokens')}")
    print(f"  Tokens saved    : {result.get('tokens_saved')}")
    print(f"  Token reduction : {_fmt(result.get('token_reduction_pct'), 2)}%")

    print(f"\n[Cost]")
    print(f"  Baseline cost   : ${_fmt(result.get('baseline_cost'), 6)}")
    print(f"  Estimated cost  : ${_fmt(result.get('estimated_cost'), 6)}")
    print(f"  Cost saved      : ${_fmt(result.get('cost_saved'), 6)}")
    print(f"  Cost reduction  : {_fmt(result.get('cost_reduction_pct'), 2)}%")
    print(f"  Budget OK       : {result.get('budget_ok')}")

    print(f"\n[Quality — {result.get('evaluator')}]")
    print(f"  Faithfulness    : {_fmt(result.get('faithfulness'))}")
    print(f"  Answer relevancy: {_fmt(result.get('answer_relevancy'))}")
    print(f"  Quality score   : {_fmt(result.get('quality_score'))}")

    print(f"\n[Latency]")
    print(f"  Total latency   : {_fmt(result.get('latency_ms'), 1)} ms")

    if result.get("answer"):
        print("\n" + "-" * 60)
        print("ANSWER")
        print("-" * 60)
        print(result["answer"])
    else:
        print("\n[No answer generated. Check status above.]")


if __name__ == "__main__":
    main()
