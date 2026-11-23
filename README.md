# Corrective RAG (CRAG) v2 — Legal Research Agent
(streamlit url: https://cragv2-fkrohmctyh2dajkhuupsen.streamlit.app/)
A deterministic, self-correcting Retrieval-Augmented Generation (CRAG) agent engineered for high-precision querying over complex statutory documents, evaluated on **India's Digital Personal Data Protection (DPDP) Rules, 2025**.

Unlike standard RAG pipelines that blindly pass noisy chunks to an LLM, this system orchestrates a stateful **LangGraph** workflow combining **hybrid search (ChromaDB + BM25 + RRF)**, **long-context cross-encoder reranking**, **strict JSON relevance grading**, **HyDE query rewriting**, **Human-in-the-Loop (HITL) verification**, and **Langfuse v5 observability**.

---

## Key Results at a Glance

* **28/28 (100%) Factual Accuracy & 0% Hallucinations** on a 28-question statutory benchmark (`final_crag_v2_benchmark_results.csv`), compared to **~11% (~3/28)** core accuracy for raw `openai/gpt-oss-120b` without retrieval.
* **100% First-Pass Retrieval (`retries_used = 0`)** across all 28 benchmark questions using hybrid BM25 + dense vector search with an 8,192-token ModernBERT cross-encoder reranker.
* **Zero-Hallucination Fallback:** Bounded self-correction loop (`max_retries = 3`) routes unanswerable or out-of-scope queries to an explicit refusal node instead of guessing.
* **Production Observability:** Instrumented with **Langfuse v5** to profile per-node latency and token consumption across LangGraph transitions and raw Groq API calls[cite: 1, 2].
* **Multi-Interface Serving:** Exposed via a **Streamlit** HITL interface, a **FastAPI** REST backend, and an **MCP (Model Context Protocol)** server sharing a single core graph.

---

## Why Standard LLMs & Naive RAG Fail on Legal Statutes

Testing a 120B frontier model (`openai/gpt-oss-120b`) directly on the DPDP Rules, 2025 exposed three recurring failure modes:

1. **Statutory & Numerical Hallucinations:** Without grounded retrieval, the raw model confidently fabricated deadlines, section numbers, and schedules (e.g., citing *30 days* instead of *45 days*, *Section 5(3)* instead of *Section 40(1)*, and *Schedule III* instead of the *Seventh Schedule*).
2. **Cross-Jurisdiction & Draft Bleed:** The raw LLM repeatedly invented non-existent regulators (*"Data Protection Authority"* instead of the *Board*) and imported GDPR penalties (*"4% of global turnover"*) and outdated 2022/2023 draft provisions.
3. **Reranker Truncation on Nested Clauses:** Standard 512-token cross-encoders (such as `ms-marco-MiniLM-L-6-v2`) degraded legal accuracy during testing because they silently truncated dense multi-clause chunks, dropping exceptions and sub-rules located deeper in the text.

---

## System Architecture

```text
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  1. retrieval_node (Hybrid Search + Long-Context Rerank) │
│  • ChromaDB Dense Search (top-10) + BM25 Keyword (top-10)│
│  • Reciprocal Rank Fusion (k=60) → top-10 candidates     │
│  • ModernBERT Cross-Encoder Reranker → top-3 chunks      │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  2. grader_node (Strict JSON Verification Gate)          │
│  • Evaluates if top-3 chunks contain the exact answer    │
│  • Enforces temperature=0.0 & JSON output schema         │
└──────────────┬───────────────────────────┬───────────────┘
               │ relevant == true          │ relevant == false
               ▼                           ▼
   [HITL Interrupt (Streamlit)]     retries < 3 ?
               │                    ├── Yes ──► 3. hyde_rewrite_node
               ▼                    │           (Rewrites query as a legal
┌──────────────────────────────┐    │            clause & loops to Step 1)
│ 4. generator_node            │    │
│ • Grounded legal synthesis   │    └── No ───► 5. failure_node
│   from verified chunks only  │                (Explicit non-hallucination
└──────────────┬───────────────┘                 refusal after 3 retries)
               ▼
         Final Answer
```

### Core Engineering Design Decisions

* **Hierarchical Legal Parsing (`loaders.py`):** Uses **Docling** (`langchain-docling`) to preserve document hierarchy, clause numbering, and page metadata when ingesting `DPDP_Rules_2025_English_only.pdf`.
* **Long-Context Reranking (`retrieval.py`):** Uses `Alibaba-NLP/gte-reranker-modernbert-base` (8,192-token window) over RRF-fused candidates so full statutory clauses are scored without truncation.
* **Deterministic & Fault-Tolerant LLM Wrapper (`langgraph_main.py`):** Enforces `temperature=0.0`, strict JSON validation on the grader, and bounded retry logic (`max_retries=3`) that reads Groq's `retry-after` header on HTTP `429` rate limits.
* **Human-in-the-Loop Checkpointing:** Compiled with LangGraph's `MemorySaver` and optional `interrupt_before=["generator_node"]` so reviewers can inspect retrieved clauses in Streamlit before generation executes.

---

## Benchmark Results (28-Question Legal Evaluation)

Evaluated on `native_synthetic_benchmark.csv` (28 multi-hop and statutory-precision questions generated from the DPDP Rules, 2025) using an automated **LLM-as-a-Judge** verifier (`evaluation.py`). Full outputs and judge verdicts are logged in `final_crag_v2_benchmark_results.csv`.

| Metric | CRAG v2 (`rag_answer`) | Raw `openai/gpt-oss-120b` (`llm_only_answer`) |
|---|---|---|
| **Factually Correct Answers** | **28 / 28 (100%)** | ~3 / 28 (~11%) *(with fabricated citations)* |
| **Hallucinations Flagged by Judge** | **0 / 28 (0%)** | ~25+ / 28 |
| **First-Pass Retrieval (`retries_used = 0`)** | **28 / 28 (100%)** | N/A |
| **Grader Relevance Pass (`is_relevant = True`)** | **28 / 28 (100%)** | N/A |
| **Direct Source Grounding (LLM Judge)** | **27 / 28 (96.4%)**\* | N/A |

*\*On Q14, the answer was factually exact and quoted the statute, but the strict verifier flagged it for paraphrasing part of the explanation rather than quoting verbatim.*

### Sample Hallucinations Eliminated vs. Raw LLM

| Question | Ground Truth (DPDP Rules, 2025) | Raw 120B LLM Output | CRAG v2 Output |
|---|---|---|---|
| **Q1** — Objection submission window | **45 days** from 3 Jan 2025 (by 17 Feb 2025) | *30 days* | **45 days** (by 17 Feb 2025) |
| **Q2** — Provision mandating draft publication | **Section 40(1)**, DPDP Act 2023 | *Section 5(3)* | **Section 40(1)** |
| **Q4** — Rules effective after 18 months | **Rules 3, 5–16, 22, and 23** | *Rules 5, 9, 13, and 14* | **Rules 3, 5–16, 22, and 23** |
| **Q24** — Clause requiring breach-causer findings | **Sub-clause (iv)** | *Clause (viii), 2022 Rules* | **Sub-clause (iv)** |
| **Q28** — Minimum retention & schedule | **1 year**; **Seventh Schedule** | *30 days; Schedule III* | **1 year; Seventh Schedule** |

---

## Observability & Performance Profiling (Langfuse v5)

The graph is instrumented inside `langgraph_main.py` using Langfuse's `CallbackHandler` (attached via `compiled_graph.with_config(...)`) and `@observe(as_type="generation")` on `call_llm()` to capture nested LangGraph node transitions, latencies, and Groq token counts without modifying the serving layers.

<!-- Drop your Langfuse trace screenshot at assets/langfuse_trace.png -->
<!-- ![Langfuse Trace Waterfall](assets/langfuse_trace.png) -->

### Empirical Bottlenecks Uncovered by Tracing

| Component | Observed p50 Latency | Key Engineering Finding |
|---|---|---|
| **`retrieval_node`** | **8.65s**[cite: 2] | Accounts for ~85% of total runtime (`9.52s` p50) due to running FP32 `gte-reranker-modernbert-base` (149M params) across 10 legal chunks on CPU[cite: 2]. |
| **`grader_node`** | **1.33s**[cite: 2] | Emitting a 5-token JSON verdict (`{"relevant": true}`) consumed ~149 output tokens due to `openai/gpt-oss-120b`'s internal reasoning tokens[cite: 1, 2]. |
| **`generator_node`** | **1.19s**[cite: 2] | Fast inference on Groq (`groq_llm_call` p50: `1.22s`)[cite: 2]. Input token volume (~850 tokens/call) is driven primarily by passing the top-3 retrieved chunks to both grader and generator[cite: 1]. |
| **`route_after_grade`** | **0.00s**[cite: 2] | Pure Python conditional routing[cite: 2]. |

### Documented Accuracy-vs-Latency Trade-off & Next Optimizations

* **Why ModernBERT was kept over faster CPU rerankers:** Switching to standard 512-token rerankers lowered retrieval latency but degraded legal benchmark accuracy due to clause truncation.
* **Production scaling path:** Serve `gte-reranker-modernbert-base` via **INT8-quantized ONNX** on CPU or **GPU with Flash Attention 2** to drop `retrieval_node` below `0.5s`, and set `reasoning_effort="low"` (or route to a smaller model) on `grader_node` to cut hidden reasoning token burn.

---

## Repository Structure

| File | Responsibility |
|---|---|
| `langgraph_main.py` | Core LangGraph CRAG state machine, Groq retry wrapper, HITL checkpointing, and Langfuse v5 tracing |
| `retrieval.py` | Hybrid retrieval (ChromaDB + BM25), Reciprocal Rank Fusion, and ModernBERT cross-encoder reranking |
| `loaders.py` | Docling-based PDF parsing and hierarchical chunking |
| `embeddings.py` | Embedding model setup, ChromaDB indexing, and BM25 index construction |
| `langgraph_app.py` | Streamlit UI with Human-in-the-Loop chunk inspection before final generation |
| `api.py` | FastAPI REST service (`/health`, `/upload_pdf`, `/query`) |
| `mcp_server.py` | Model Context Protocol (MCP) server exposing the CRAG agent as an external tool |
| `generate_testset.py` | Synthetic legal QA test set generator |
| `evaluation.py` | Automated benchmark runner comparing Raw LLM vs. CRAG v2 with LLM-as-a-Judge verification |
| `DPDP_Rules_2025_English_only.pdf` | Source statutory document (India's DPDP Rules, 2025) |
| `native_synthetic_benchmark.csv` | 28-question evaluation dataset |
| `final_crag_v2_benchmark_results.csv` | Full benchmark outputs, retry counts, and LLM-judge verdicts |
| `requirements.txt` | Top-level Python dependencies |
| `.env.example` | Environment variable template for Groq and optional Langfuse tracing |

---

## Quickstart

### 1. Prerequisites
* **Python 3.11+** (required for `langfuse>=5.0.0`)
* A **Groq API Key**
* *(Optional)* A free **Langfuse Cloud** account for trace observability

### 2. Installation & Environment Setup
```bash
git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
cd <your-repo-name>

pip install -r requirements.txt
```

Create a `.env` file in the project root based on `.env.example`:
```env
# Required
GROQ_API_KEY=gsk_your_groq_api_key

# Optional — Langfuse v5 Observability (app runs cleanly without these)
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_BASE_URL=[https://cloud.langfuse.com](https://cloud.langfuse.com)
```

### 3. Run the Interfaces

**Streamlit HITL Web App:**
```bash
streamlit run langgraph_app.py
```

**FastAPI Backend:**
```bash
uvicorn api:app --reload
```
*Endpoints:* `GET /health` · `POST /upload_pdf` · `POST /query` (interactive Swagger docs at `http://localhost:8000/docs`)

**Run the 28-Question Benchmark:**
```bash
python evaluation.py
```

**Run the MCP Server:**
```bash
python mcp_server.py
```

---

## Tech Stack

* **Orchestration & Agents:** LangGraph, LangChain Core
* **LLM Inference:** Groq API (`openai/gpt-oss-120b`)
* **Parsing & Retrieval:** Docling (`langchain-docling`), ChromaDB, `rank-bm25`, Sentence-Transformers (`Alibaba-NLP/gte-reranker-modernbert-base`)
* **Observability:** Langfuse v5 (OpenTelemetry)
* **Serving & Interfaces:** Streamlit, FastAPI, Uvicorn, Pydantic, MCP
* **Evaluation:** Pandas, Automated LLM-as-a-Judge
