"""
Benchmark Report
Runs a list of test queries through the gateway, collects metrics, and prints
a comparison table (Baseline RAG Agent vs Optimised Gateway).

Usage:
    python scripts/benchmark_report.py
    python scripts/benchmark_report.py --queries my_queries.txt  (one question per line)
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from efficiency_gateway.core.metrics_store import MetricsStore
from efficiency_gateway.workflows.rag_gateway_graph import build_gateway_graph

DEFAULT_QUERIES = [
    "What is attention mechanism in transformers?",
    "How does LoRA reduce fine-tuning parameters?",
    "Explain RLHF in large language models.",
    "What are the limitations of RAG systems?",
    "Compare BM25 and dense retrieval methods.",
]

SEPARATOR = "-" * 70


def _pct(v) -> str:
    return f"{v:.1f}%" if v is not None else "—"


def _cost(v) -> str:
    return f"${v:.6f}" if v is not None else "—"


def _tok(v) -> str:
    return str(v) if v is not None else "—"


def run_benchmark(queries: list[str]) -> list[dict]:
    app = build_gateway_graph(
        persist_dir=str(ROOT / "data" / "chroma"),
        collection_name="research_papers",
        router_config_path=str(ROOT / "configs" / "model_router.json"),
        pricing_path=str(ROOT / "configs" / "pricing.json"),
        db_path=str(ROOT / "data" / "metrics.db"),
    )

    results = []
    for i, q in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {q[:60]}...")
        t0 = time.perf_counter()
        state = app.invoke({"question": q, "required_quality": "normal"})
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        state["_wall_ms"] = elapsed
        results.append(state)

    return results


def print_report(results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("  LLM EFFICIENCY GATEWAY — BENCHMARK REPORT")
    print("=" * 70)

    rows = []
    for r in results:
        rows.append({
            "question": r.get("question", "")[:55],
            "model": r.get("selected_model", ""),
            "baseline_in": r.get("baseline_input_tokens", 0),
            "opt_in": r.get("actual_input_tokens", 0),
            "tok_red": r.get("token_reduction_pct", 0.0),
            "baseline_cost": r.get("baseline_cost", 0.0),
            "opt_cost": r.get("estimated_cost", 0.0),
            "cost_red": r.get("cost_reduction_pct", 0.0),
            "quality": r.get("quality_score", 0.0),
            "latency": r.get("latency_ms", r.get("_wall_ms", 0.0)),
            "cache": r.get("cache_hit", False),
        })

    # per-query table
    hdr = (
        f"{'Question':<55}  {'Model':<15}  "
        f"{'Tok↓%':>6}  {'Cost↓%':>7}  {'Quality':>7}  {'Latency':>9}"
    )
    print(hdr)
    print(SEPARATOR)
    for r in rows:
        cache_tag = "[CACHE] " if r["cache"] else ""
        print(
            f"{cache_tag}{r['question']:<55}  "
            f"{r['model']:<15}  "
            f"{_pct(r['tok_red']):>6}  "
            f"{_pct(r['cost_red']):>7}  "
            f"{r['quality']:.3f}  "
            f"{r['latency']:>8.0f}ms"
        )

    # aggregate
    non_cache = [r for r in rows if not r["cache"]]
    if non_cache:
        n = len(non_cache)
        avg_baseline_in = sum(r["baseline_in"] for r in non_cache) / n
        avg_opt_in = sum(r["opt_in"] for r in non_cache) / n
        avg_tok_red = sum(r["tok_red"] for r in non_cache) / n
        avg_baseline_cost = sum(r["baseline_cost"] for r in non_cache) / n
        avg_opt_cost = sum(r["opt_cost"] for r in non_cache) / n
        avg_cost_red = sum(r["cost_red"] for r in non_cache) / n
        avg_quality = sum(r["quality"] for r in non_cache) / n
        avg_latency = sum(r["latency"] for r in non_cache) / n

        print("\n" + "=" * 70)
        print("  AGGREGATE (LLM calls only, excluding cache hits)")
        print("=" * 70)
        header = f"{'Metric':<35}  {'Baseline':>15}  {'Optimised':>15}"
        print(header)
        print(SEPARATOR)
        print(f"{'Avg input tokens':<35}  {avg_baseline_in:>15.0f}  {avg_opt_in:>15.0f}")
        print(f"{'Estimated cost / request':<35}  {avg_baseline_cost:>15.6f}  {avg_opt_cost:>15.6f}")
        print(f"{'Avg token reduction':<35}  {'—':>15}  {avg_tok_red:>14.1f}%")
        print(f"{'Avg cost reduction':<35}  {'—':>15}  {avg_cost_red:>14.1f}%")
        print(f"{'Avg quality score':<35}  {'—':>15}  {avg_quality:>15.3f}")
        print(f"{'Avg latency (ms)':<35}  {'—':>15}  {avg_latency:>14.0f}")

    cache_hits = sum(1 for r in rows if r["cache"])
    print(f"\nCache hit rate: {cache_hits}/{len(rows)} = {cache_hits/len(rows)*100:.1f}%")

    # save JSON
    out_path = ROOT / "data" / "benchmark_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"\nFull report saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run LLM Efficiency Gateway benchmark")
    parser.add_argument("--queries", type=str, default=None, help="Path to a .txt file with one query per line")
    args = parser.parse_args()

    if args.queries:
        queries = Path(args.queries).read_text(encoding="utf-8").splitlines()
        queries = [q.strip() for q in queries if q.strip()]
    else:
        queries = DEFAULT_QUERIES

    print(f"Running {len(queries)} queries through the gateway...\n")
    results = run_benchmark(queries)
    print_report(results)


if __name__ == "__main__":
    main()
