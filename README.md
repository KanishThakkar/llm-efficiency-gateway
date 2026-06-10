# ⚡ LLM Efficiency Gateway

> An AI middleware system that reduces LLM API cost, latency, and token usage using **semantic caching**, **hybrid RAG retrieval**, **prompt compression**, **quality-aware model routing**, and **security guardrails** while preserving answer quality.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Workflow-green)
![RAG](https://img.shields.io/badge/RAG-Hybrid%20Search-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

---

# 🚀 Key Results

| Metric                  | Value     |
| ----------------------- | --------- |
| Average Cost Reduction  | **58.1%** |
| Peak Cost Reduction     | **90.9%** |
| Cache Hit Response Time | **12ms**  |
| Cache Hit Rate          | **31.2%** |
| Average LLM Latency     | **~1.6s** |
| Semantic Cache          | ✅         |
| Hybrid RAG              | ✅         |
| Model Routing           | ✅         |
| Prompt Compression      | ✅         |
| Security Guardrails     | ✅         |

---

# 🎯 Problem Statement

Modern AI applications often send every request to expensive large language models regardless of complexity.

This leads to:

* High API costs
* Increased latency
* Token wastage
* Oversized RAG contexts
* Poor resource utilization

The **LLM Efficiency Gateway** acts as an optimization layer between users and LLMs, reducing cost and latency without significantly degrading answer quality.

---

# 🏗️ System Architecture

```text
User Query
      │
      ▼
Prompt Injection + PII Guard
      │
      ▼
Token Counter + Cost Estimator
      │
      ▼
Semantic Cache (ChromaDB)

 ┌─────────────────────────┐
 │       Cache Hit         │
 └──────────┬──────────────┘
            ▼
     Return Cached Answer
            ▼
      Cost Tracker

 ┌─────────────────────────┐
 │      Cache Miss         │
 └──────────┬──────────────┘
            ▼
Hybrid Retrieval (BM25 + Vector)
            ▼
Retrieved Context Security Check
            ▼
CrossEncoder Reranker
            ▼
Adaptive Context Pruning
            ▼
Prompt Compression
            ▼
Quality-Aware Model Router
            ▼
Answer Generation
            ▼
Semantic Cache Update
            ▼
Quality Evaluation
            ▼
Metrics Storage (SQLite)
            ▼
Streamlit Dashboard
```

---

# 🔥 Features

## Semantic Cache

* ChromaDB-powered semantic cache
* Reuses answers for semantically similar queries
* Eliminates unnecessary LLM calls
* Average cache response time: **12ms**

---

## Hybrid Retrieval

Combines:

* BM25 lexical search
* Dense vector retrieval
* HuggingFace embeddings

Provides better recall than either retrieval method alone.

---

## CrossEncoder Reranking

Uses a CrossEncoder model to:

* Re-rank retrieved chunks
* Select only highly relevant context
* Improve retrieval precision

---

## Adaptive Context Pruning

Reduces token usage by:

* Removing low-value chunks
* Enforcing token budgets
* Prioritizing high-relevance context

---

## Prompt Compression

Uses **LLMLingua-2** to:

* Compress prompts
* Remove redundant instructions
* Reduce token consumption

Average compression: **~45%**

---

## Quality-Aware Model Routing

Automatically selects the most cost-effective model.

### Simple Queries

```text
groq/llama-3.1-8b-instant
```

### Complex Queries

```text
groq/llama-3.3-70b-versatile
```

This significantly lowers overall inference cost.

---

## Security Guardrails

Protects the system against:

* Prompt injection attacks
* PII leakage
* Secret exposure
* Malicious retrieval content

---

## Cost Tracking

Tracks:

* Input tokens
* Output tokens
* Estimated API cost
* Cost savings
* Latency

No paid API usage required for benchmarking.

---

## Quality Evaluation

Evaluates generated answers using:

* Faithfulness
* Answer Relevancy
* Text Similarity Metrics

Ensures optimization does not excessively reduce answer quality.

---

# 📊 Benchmark Results

| Metric                   | Baseline  | Optimized Gateway |
| ------------------------ | --------- | ----------------- |
| Avg Input Tokens         | 1,704     | 1,788             |
| Estimated Cost / Request | $0.001140 | $0.000136         |
| Cost Reduction           | —         | **90.9% Peak**    |
| Average Cost Reduction   | —         | **58.1%**         |
| Cache Hit Rate           | —         | **31.2%**         |
| Cache Response Time      | —         | **12ms**          |
| LLM Response Time        | —         | **~1.6s**         |

---

# 🛠️ Tech Stack

| Layer          | Technology                      |
| -------------- | ------------------------------- |
| Workflow       | LangGraph, LangChain            |
| LLM Provider   | Groq API                        |
| Model Routing  | LiteLLM                         |
| Vector Store   | ChromaDB                        |
| Retrieval      | BM25 + Dense Retrieval          |
| Embeddings     | HuggingFace                     |
| Reranking      | CrossEncoder                    |
| Compression    | LLMLingua-2                     |
| Evaluation     | Custom Faithfulness + Relevancy |
| Security       | Regex-based Guardrails          |
| Metrics        | SQLite                          |
| Dashboard      | Streamlit + Plotly              |
| Token Counting | tiktoken                        |

---

# 📁 Project Structure

```text
llm-efficiency-gateway/
│
├── configs/
│   ├── model_router.json
│   └── pricing.json
│
├── data/
│   └── papers/
│
├── dashboard/
│   └── app.py
│
├── scripts/
│   ├── ingest_papers_step2.py
│   ├── run_workflow_step10.py
│   ├── benchmark_report.py
│   └── diagnose.py
│
├── src/
│   └── efficiency_gateway/
│       ├── cache/
│       ├── core/
│       ├── evaluation/
│       ├── llm/
│       ├── rag/
│       ├── routing/
│       ├── security/
│       └── workflows/
│
└── requirements.txt
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/KanishThakkar/llm-efficiency-gateway.git
cd llm-efficiency-gateway
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Configure API Key

### Windows PowerShell

```powershell
$env:GROQ_API_KEY="your-api-key"
```

### Linux / Mac

```bash
export GROQ_API_KEY="your-api-key"
```

---

# 📚 Ingest Research Papers

Place PDFs inside:

```text
data/papers/
```

Recommended papers:

* Attention Is All You Need
* Retrieval-Augmented Generation (RAG)
* LoRA
* Quantization

Run:

```bash
python scripts/ingest_papers_step2.py
```

---

# ▶️ Run Workflow

```bash
python scripts/run_workflow_step10.py
```

---

# 🧪 Benchmark

```bash
python scripts/benchmark_report.py
```

---

# 📈 Launch Dashboard

```bash
streamlit run dashboard/app.py
```

---

# 📊 Dashboard Features

The Streamlit dashboard provides:

* KPI Cards
* Cost Savings Analysis
* Cache Hit Metrics
* Cost–Quality Pareto Frontier
* Model Usage Distribution
* Latency Analysis
* Token Savings Trends
* Historical Query Runs

---

# 💡 Design Decisions

### LiteLLM

Provider-agnostic model abstraction.

Switch between:

* Groq
* OpenAI
* Anthropic
* Ollama

with minimal code changes.

### LangGraph

Provides:

* Conditional routing
* State management
* Cache hit/miss branching

### ChromaDB

Used for:

* Vector retrieval
* Semantic caching

Single persistent storage solution.

### SQLite

Chosen for:

* Simplicity
* Zero infrastructure
* Portability

---

