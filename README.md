# ⚡ LLM Efficiency Gateway

An AI middleware system that reduces LLM API cost and latency using semantic caching, 
hybrid RAG retrieval, quality-aware model routing, prompt compression, and security 
guardrails — while preserving answer quality.

> **Result:** 58.1% average cost reduction | 90.9% peak cost reduction | 12ms cache response | 31.2% cache hit rate

---

## 🎯 What It Does

Modern AI applications waste money by sending every request to expensive large models.
This gateway sits between your application and the LLM, intelligently optimizing every request.

| Component | What It Does |
|---|---|
| **Semantic Cache** | Reuses previous answers for similar queries — zero LLM cost |
| **Hybrid Retrieval** | BM25 + vector search for best context retrieval |
| **CrossEncoder Reranker** | Picks the most relevant chunks before sending to LLM |
| **Context Pruner** | Stays within token budget, removes low-value chunks |
| **Prompt Compression** | LLMLingua-2 compresses prompts by ~45% |
| **Model Router** | Routes simple queries to cheap models, complex ones to strong models |
| **Security Guard** | Blocks prompt injection, masks PII, detects secrets |
| **Cost Tracker** | Estimates and tracks API cost per request |
| **Quality Evaluator** | Measures faithfulness and answer relevancy |
| **Dashboard** | Real-time Streamlit dashboard with Pareto charts |

---

## 🔄 Pipeline
User Query
↓ Prompt Injection + PII Guard
↓ Token Counter + Cost Estimator
↓ Semantic Cache (ChromaDB)
├── Cache Hit → Return Answer (12ms, $0.00)
└── Cache Miss ↓
↓ Hybrid Search: BM25 + Vector Retrieval
↓ Retrieved Context Security Check
↓ CrossEncoder Reranker
↓ Adaptive Context Pruning
↓ Prompt Compression (LLMLingua-2)
↓ Quality-Aware Model Router
├── Simple query → groq/llama-3.1-8b-instant (cheap)
└── Complex query → groq/llama-3.3-70b-versatile (strong)
↓ Answer Generation
↓ Store to Semantic Cache
↓ RAGAS Quality Evaluation
↓ Cost + Token Savings Computed
↓ SQLite Metrics Storage
↓ Streamlit Dashboard

---

## 📊 Benchmark Results

| Metric | Baseline | Optimised |
|---|---|---|
| Avg input tokens | 1,704 | 1,788 |
| Estimated cost / request | $0.001140 | $0.000136 |
| Cost reduction | — | **90.9% (peak)** |
| Avg cost reduction | — | **58.1%** |
| Cache latency | — | **12ms** |
| LLM latency | — | ~1.6s |
| Cache hit rate | — | 31.2% |

---

## 🛠️ Tech Stack

| Layer | Stack |
|---|---|
| Workflow | LangGraph, LangChain |
| LLM | Groq API (llama-3.1-8b-instant, llama-3.3-70b-versatile) via LiteLLM |
| RAG | ChromaDB, BM25, HuggingFace Embeddings, CrossEncoder |
| Compression | LLMLingua-2 |
| Security | Custom regex guard (prompt injection, PII, secrets) |
| Evaluation | Text-overlap (faithfulness + relevancy) |
| Token Counting | tiktoken |
| Storage | SQLite |
| Dashboard | Streamlit + Plotly |

---

## 🚀 Getting Started

### 1. Clone the repo
``bash
git clone https://github.com/KanishThakkar/llm-efficiency-gateway.git
cd llm-efficiency-gateway

2. Install dependencies
pip install -r requirements.txt
3. Set your Groq API key
 # Get free API key from https://console.groq.com
$env:GROQ_API_KEY = "your-groq-api-key-here"   # PowerShell
export GROQ_API_KEY="your-groq-api-key-here"    # Linux/Mac
4. Add PDF papers and ingest
# Add PDF files to data/papers/
# Recommended: download from arxiv.org
#   - Attention Is All You Need
#   - RAG paper 
#   - LoRA
#   - Quantization
python scripts/ingest_papers_step2.py

5. Run diagnostics (verify everything works)
python scripts/diagnose.py

6. Run a query
python scripts/run_workflow_step10.py

7. Run benchmark
python scripts/benchmark_report.py

8. Launch dashboard
streamlit run dashboard/app.py

📁 Project Structure
llm-efficiency-gateway/
├── src/efficiency_gateway/
│   ├── workflows/
│   │   └── rag_gateway_graph.py      # LangGraph pipeline
│   ├── security/
│   │   └── input_guard.py            # Prompt injection + PII guard
│   ├── cache/
│   │   └── semantic_cache.py         # ChromaDB semantic cache
│   ├── rag/
│   │   ├── hybrid_retriever.py       # BM25 + vector hybrid search
│   │   ├── reranker.py               # CrossEncoder reranking
│   │   ├── context_pruner.py         # Adaptive token pruning
│   │   ├── prompt_compressor.py      # LLMLingua-2 compression
│   │   └── prompt_builder.py         # RAG prompt construction
│   ├── routing/
│   │   └── model_router.py           # Quality-aware model routing
│   ├── llm/
│   │   └── answer_generator.py       # LiteLLM answer generation
│   ├── evaluation/
│   │   └── ragas_evaluator.py        # Quality evaluation
│   └── core/
│       ├── token_counter.py          # tiktoken token counting
│       ├── cost_tracker.py           # Cost estimation
│       └── metrics_store.py          # SQLite metrics storage
├── scripts/
│   ├── run_workflow_step10.py        # Single query runner
│   ├── benchmark_report.py           # Benchmark 5 queries
│   ├── ingest_papers_step2.py        # Ingest PDF papers
│   └── diagnose.py                   # Component health check
├── dashboard/
│   └── app.py                        # Streamlit dashboard
├── configs/
│   ├── model_router.json             # Model tier configuration
│   └── pricing.json                  # API pricing config
├── data/
│   └── papers/                       # Add your PDF papers here
└── requirements.txt

📈 Dashboard
The Streamlit dashboard shows:

KPI cards — total runs, cache hit rate, avg cost reduction, avg latency
Cost-Quality Pareto — cost vs quality per model (bubble size = tokens saved)
Token Savings chart — baseline vs optimised input tokens per run
Model Usage pie — routing distribution across models
Latency histogram — cache hits vs LLM calls
Quality by model — box plot comparing model quality scores
Recent runs table — full history with all metrics

You can see the streamlit web app screenshot - 
<img width="1900" height="876" alt="image" src="https://github.com/user-attachments/assets/999b303c-0aa9-45ce-8439-bc0d4abb445d" />
<img width="1898" height="703" alt="image" src="https://github.com/user-attachments/assets/95b3ec10-5f17-418d-8409-dd611daa4c29" />
<img width="2076" height="930" alt="image" src="https://github.com/user-attachments/assets/84fcdb49-1723-4d26-b5d3-b54b63f720b5" />




💡 Key Design Decisions
LiteLLM for model abstraction — swap Groq for OpenAI/Anthropic/Ollama with one config change
LangGraph for workflow — conditional edges enable clean cache-hit/miss branching
ChromaDB for both vector store and semantic cache — single dependency, persistent storage
Regex-based security guard — no heavy ML models required, works on any machine
SQLite for metrics — zero infrastructure, fully portable

🔧 Configuration
Switch models — edit configs/model_router.json:

{
  "defaults": {
    "cheap_model": "groq/llama-3.1-8b-instant",
    "strong_model": "groq/llama-3.3-70b-versatile",
    "local_model": "local-quantized"
  }
}
Adjust pricing — edit configs/pricing.json to match your provider's rates.
