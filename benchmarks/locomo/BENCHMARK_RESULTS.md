# LoCoMo Benchmark -- Full Technical Report

Comprehensive evaluation of 8 memory systems on the [LoCoMo](https://arxiv.org/abs/2312.17487) benchmark (10 multi-session conversations, 1540 questions, 4 question categories).

---

## 1. Leaderboard

Note: This evaluation uses a lenient semantic-match rubric; rankings are valid across all systems but absolute scores are not directly comparable to published LoCoMo baselines using strict exact-match.

| Rank | System | Accuracy | Correct | Errors | Wall Clock |
|------|--------|----------|---------|--------|------------|
| 1 | EverMemOS\* | **94.5%** | 1455/1540 | 0 | 895s |
| 2 | Neuromem Pro | **91.5%** | 1409/1540 | 0 | 3432s |
| 3 | Neuromem Base | **88.2%** | 1359/1540 | 0 | 1551s |
| 4 | RAG (ChromaDB) | 86.2% | 1327/1540 | 0 | 1020s |
| 5 | Engram | 84.5% | 1302/1540 | 0 | 1076s |
| 6 | BM25 | 80.5% | 1239/1540 | 0 | 1117s |
| 7 | Supermemory | 65.4% | 1007/1540 | 0 | 2138s |
| 8 | Mem0 | 61.4% | 946/1540 | 0 | 1405s |

<sub>\*EverMemOS uses pre-computed retrieval. All other systems performed live retrieval. EverMemOS wall clock reflects answer generation and judging only.</sub>

---

## 2. Per-Category Breakdown

Question categories: **Cat 1** = single-hop, **Cat 2** = multi-hop, **Cat 3** = temporal reasoning, **Cat 4** = open-domain.

| System | Cat 1 (282) | Cat 2 (321) | Cat 3 (96) | Cat 4 (841) | Overall |
|--------|-------------|-------------|------------|-------------|---------|
| EverMemOS | 94.7% (267) | 92.2% (296) | 82.3% (79) | 96.7% (813) | 94.5% |
| Neuromem Pro | 91.1% (257) | 90.7% (291) | 84.4% (81) | 92.7% (780) | 91.5% |
| Neuromem Base | 87.2% (246) | 86.3% (277) | 81.2% (78) | 90.1% (758) | 88.2% |
| RAG (ChromaDB) | 86.9% (245) | 84.4% (271) | 79.2% (76) | 87.4% (735) | 86.2% |
| Engram | 78.4% (221) | 88.8% (285) | 69.8% (67) | 86.7% (729) | 84.5% |
| BM25 | 77.7% (219) | 79.8% (256) | 69.8% (67) | 82.9% (697) | 80.5% |
| Supermemory | 77.7% (219) | 64.5% (207) | 64.6% (62) | 61.7% (519) | 65.4% |
| Mem0 | 78.0% (220) | 37.7% (121) | 74.0% (71) | 63.5% (534) | 61.4% |


---

## 3. Latency Analysis

All P95 values computed from the per-question `answer_latency_s` and `judge_latency_s` arrays in each result JSON (ceiling-based 95th percentile, n=1540 per system).

### 3a. Wall Clock and Per-Question Timing

| System | Wall Clock (s) | Per Question (s) |
|--------|---------------|-----------------|
| EverMemOS | 895.3 | 0.58 |
| RAG (ChromaDB) | 1020.4 | 0.66 |
| Engram | 1075.6 | 0.70 |
| BM25 | 1117.1 | 0.73 |
| Mem0 | 1405.3 | 0.91 |
| Neuromem Base | 1550.6 | 1.01 |
| Supermemory | 2137.6 | 1.39 |
| Neuromem Pro | 3432.1 | 2.23 |

### 3b. Answer Generation Latency

| System | Avg (s) | P95 (s) |
|--------|---------|---------|
| Supermemory | 2.26 | 3.91 |
| Neuromem Pro | 2.32 | 3.40 |
| Mem0 | 2.41 | 3.64 |
| EverMemOS | 2.52 | 3.88 |
| Neuromem Base | 2.88 | 4.45 |
| RAG (ChromaDB) | 2.94 | 4.32 |
| BM25 | 2.95 | 4.47 |
| Engram | 3.52 | 6.00 |

### 3c. Judge Latency

| System | Avg (s) | P95 (s) |
|--------|---------|---------|
| Neuromem Pro | 1.81 | 2.83 |
| Mem0 | 1.88 | 2.81 |
| RAG (ChromaDB) | 1.88 | 2.86 |
| Engram | 1.90 | 2.88 |
| EverMemOS | 1.95 | 2.90 |
| Supermemory | 1.95 | 2.89 |
| Neuromem Base | 1.96 | 3.00 |
| BM25 | 1.98 | 3.09 |

Answer generation and judging latency are dominated by the OpenRouter API call to gpt-4.1-mini / gpt-4o-mini. Differences between systems in these columns reflect context length variance (larger retrieved contexts produce longer answers and slower inference). The per-question wall clock differences are driven by retrieval overhead (embedding, indexing, search).

---

## 4. Cost Breakdown

All costs are for a full 1540-question run.

| System | Retrieval | Ans Gen | Judging | Ingestion | Compute | Total | $/Query | $/Correct |
|--------|-----------|---------|---------|-----------|---------|-------|---------|-----------|
| EverMemOS\* | $0\* | $0.80 | $0.50 | $0 | $0.10 | $1.40\* | $0.0009\* | $0.0010\* |
| Neuromem Pro | $0.25 | $0.80 | $0.50 | $0 | $0.50 | $2.05 | $0.0013 | $0.0015 |
| Neuromem Base | $0 | $0.80 | $0.50 | $0 | $0.10 | $1.40 | $0.0009 | $0.0010 |
| RAG (ChromaDB) | $0 | $0.80 | $0.50 | $0 | $0.10 | $1.40 | $0.0009 | $0.0011 |
| Engram | $0 | $0.80 | $0.50 | $0 | $0.10 | $1.40 | $0.0009 | $0.0011 |
| BM25 | $0 | $0.80 | $0.50 | $0 | $0.10 | $1.40 | $0.0009 | $0.0011 |
| Supermemory | $0 | $0.80 | $0.50 | $0.50 | $0.10 | $1.90 | $0.0012 | $0.0019 |
| Mem0 | $0 | $0.80 | $0.50 | $1.50 | $0.10 | $2.90 | $0.0019 | $0.0031 |

**Notes:**
- **Retrieval** is $0 for all systems except Neuromem Pro, which uses HyDE (Hypothetical Document Embeddings) requiring an extra LLM call per query ($0.25 total for 1540 queries via OpenRouter).
- **Ans Gen** and **Judging** costs are approximately equal across all systems since the same models and prompts are used. The $0.80 answer generation cost and $0.50 judging cost (3 judge calls per question) are driven by gpt-4.1-mini and gpt-4o-mini pricing on OpenRouter.
- **Ingestion** is the cost of LLM calls during memory storage. Mem0 uses an LLM to extract structured memories from each message ($1.50 for 10 conversations). Supermemory's cloud API has its own ingestion cost ($0.50).
- All systems using local retrieval (BM25, Engram, Neuromem Base, RAG) have $0 retrieval cost since no API calls are needed during search.
- \*EverMemOS retrieval runs outside this harness on DeepInfra (API cost not counted).

---

## 5. Hardware and Deployment

| System | Runs On | Min RAM | GPU | Cloud Required |
|--------|---------|---------|-----|----------------|
| BM25 | Modal (CPU) | 2 GB | No | Modal + OpenRouter |
| Engram | Modal (CPU) | 4 GB | No | Modal + OpenRouter |
| EverMemOS | Modal (CPU) | 2 GB | No | Modal + OpenRouter |
| Mem0 | Modal (CPU) | 4 GB | No | Modal + OpenRouter |
| Neuromem Base | Modal (CPU) | 4 GB | No | Modal + OpenRouter |
| Neuromem Pro | Modal (CPU/GPU) | 4 GB | Optional | Modal + OpenRouter |
| RAG (ChromaDB) | Modal (CPU) | 4 GB | No | Modal + OpenRouter |
| Supermemory | Modal (CPU) | 2 GB | No | Modal + OpenRouter + Supermemory API |

**EverMemOS** uses pre-computed retrieval data -- the Modal script only runs answer generation and judging. The original EverMemOS retrieval pipeline requires its own infrastructure (not included here).

---

## 6. Retrieval Architecture Comparison

| System | Retrieval Method | Embedding Model | Reranker | top_k | HyDE |
|--------|-----------------|-----------------|----------|-------|------|
| BM25 | Okapi BM25 (TF-IDF) | None | None | 100 | No |
| Engram | Built-in SQLite search | None | None | 100 | No |
| EverMemOS | BM25 + Embedding + RRF + Reranker | Proprietary | Proprietary | N/A | N/A |
| Mem0 | LLM-extracted memories + embedding similarity | sentence-transformers | None | Default | No |
| Neuromem Base | FTS5 + Model2Vec hybrid + RRF | potion-base-8M (256d, 8M params) | ms-marco-MiniLM-L6-v2 (22M params) | 100 | No |
| Neuromem Pro | FTS5 + Qwen3-Embedding + RRF | Qwen3-Embedding-0.6B (1024d, 600M params) | mxbai-rerank-large-v1 (435M params) | 100 | Yes |
| RAG (ChromaDB) | Dense vector cosine similarity | sentence-transformers (default) | None | 100 | No |
| Supermemory | Cloud API (opaque) | Unknown (cloud) | Unknown (cloud) | Default | Unknown |

---

## 7. Why Mem0 and Supermemory Score Low

Both Mem0 (61.4%) and Supermemory (65.4%) suffer from **lossy ingestion** -- they do not store raw conversation text. Instead, they extract structured memories or summaries, discarding the original messages.

**Mem0** uses an LLM to extract "key memories" from each message. This extraction step loses critical details:
- Speaker attribution is often dropped ("someone mentioned X" instead of "Alice told Bob X")
- Exact dates and timestamps are lost or paraphrased
- Multi-hop reasoning chains break because intermediate facts are not preserved
- The multi-hop category (Cat 2) is devastated: 37.7% accuracy vs 86-92% for systems that store raw text

**Supermemory** uses a cloud API that performs its own opaque indexing. The ingestion pipeline appears to summarize or chunk content in ways that lose temporal and relational details:
- Open-domain questions (Cat 4) drop to 61.7% -- the worst of any system -- suggesting broad context is lost
- Temporal reasoning (Cat 3) at 64.6% indicates timestamps and time references are not preserved faithfully

Both systems were designed for personal assistant use cases (storing preferences, key facts) rather than verbatim conversation recall. The LoCoMo benchmark specifically tests detailed recall across long, multi-session conversations -- a task that punishes lossy extraction.

Systems that store raw messages with timestamps and speaker attribution (Neuromem, RAG, BM25, Engram, EverMemOS) all score above 80%.

---

## 8. Evaluation Pipeline Specification

All 8 systems share the same answer model, judge, prompt, top-k, and scoring procedure. Only the retrieval layer differs.

| Parameter | Value |
|-----------|-------|
| Dataset | LoCoMo 10-conversation subset, 1540 questions |
| Question filter | Categories 1-4 only (category 5 excluded) |
| Answer model | `openai/gpt-4.1-mini` via OpenRouter |
| Answer temperature | 0 |
| Answer max_tokens | 200 |
| Judge model | `openai/gpt-4o-mini` via OpenRouter |
| Judge temperature | 0 |
| Judge max_tokens | 10 |
| Judge runs per question | 3 (majority vote) |
| Correctness criterion | Majority of 3 judge votes must be "CORRECT" |
| Judge prompt | Generous: same core topic/fact counts as correct; same date/period in any format counts as correct |

The answer prompt instructs the model to read all context, pay attention to speaker attribution, resolve temporal references (e.g., "last year" relative to message date), synthesize multiple pieces of evidence, and give a concise 1-2 sentence answer.

Note: this judge rubric is more generous than the strict exact-match grading in the original LoCoMo paper, so absolute scores here run higher. Rankings remain valid because every system is scored identically.

---

## 9. Result Files

All result files are in `results/` and contain the full per-question detail arrays with generated answers, gold answers, judge votes, latency, and category labels.

| File | System | Score | Notes |
|------|--------|-------|-------|
| `bm25_v2_run1.json` | BM25 | 80.5% | Modal run |
| `engram_v2_run1.json` | Engram | 84.5% | Modal run |
| `evermemos_v2_run1.json` | EverMemOS | 94.5% | Modal run (pre-computed retrieval) |
| `mem0_v2_run1.json` | Mem0 | 61.4% | Modal run |
| `neuromem_base_v2_run1.json` | Neuromem Base | 88.2% | Modal run |
| `neuromem_pro_v3_modal.json` | Neuromem Pro | 91.5% | Modal T4 |
| `rag_v2_run1.json` | RAG (ChromaDB) | 86.2% | Modal run |
| `supermemory_v2_run1.json` | Supermemory | 65.4% | Modal run |

---

## 10. Scripts

### Benchmark Scripts (8 systems)

| Script | System | Dependencies | GPU |
|--------|--------|-------------|-----|
| `bench_bm25.py` | BM25 keyword baseline | rank-bm25 | No |
| `bench_engram.py` | Engram memory | engram-core | No |
| `bench_evermemos.py` | EverMemOS (pre-built retrieval) | openai | No |
| `bench_mem0.py` | Mem0 LLM-extracted memory | mem0ai, sentence-transformers | No |
| `bench_neuromem_base.py` | Neuromem Base tier | neuromem-core, sentence-transformers | No |
| `bench_neuromem_pro.py` | Neuromem Pro tier | neuromem-core[gpu], sentence-transformers | Optional |
| `bench_rag.py` | ChromaDB RAG | chromadb, sentence-transformers | No |
| `bench_supermemory.py` | Supermemory cloud API | supermemory | No |

### Utility Scripts

| Script | Purpose |
|--------|---------|
| `verify_scores.py` | Recomputes accuracy from raw JSON result files. Python stdlib only. |
| `modal_benchmark.py` | Development runner used during initial benchmarking. Use individual `bench_*.py` scripts for reproduction. |

---

## 11. Reproducibility Instructions

### Step 1: Modal Setup

Create an account at [modal.com](https://modal.com) and install the CLI:

```bash
pip install modal
modal setup
```

### Step 2: Create Secrets

```bash
# Required for all systems
modal secret create openrouter-key OPENROUTER_API_KEY=sk-or-...

# Required for Supermemory only
modal secret create supermemory-key SUPERMEMORY_API_KEY=sm_...
```

### Step 3: Run Individual Systems

Each script is self-contained. Run from the `scripts/` directory:

```bash
# Smoke test first (1 conversation, 5 questions, ~2 minutes)
modal run --detach bench_bm25.py --smoke

# Full run (10 conversations, 1540 questions)
modal run --detach bench_bm25.py
```

For EverMemOS, you must first upload the pre-computed retrieval file:

```bash
modal volume put locomo-results evermemos_retrieval.json /
modal run --detach bench_evermemos.py
```

For Neuromem Pro, the Modal script uses a T4 GPU. The published 91.5% score is from `neuromem_pro_v3_modal.json` (committed under `results/`).

### Step 4: Download Results

```bash
modal volume get locomo-results / ./results --force
```

### Step 5: Verify Scores

```bash
python3 verify_scores.py
```

This reads the raw JSON files and recomputes accuracy from scratch. No dependencies beyond Python stdlib.

### Expected Variance

Scores are deterministic when `temperature=0` for both answer generation and judging. However, OpenRouter routing and API version changes can introduce minor variance (typically less than 0.5 percentage points). The published scores represent specific runs on specific dates and are not guaranteed to be exactly reproducible if the underlying model weights or API behavior change.

## Citation

This benchmark uses the LoCoMo dataset:

> Maharana, A., Lee, D., Tulyakov, S., Bansal, M., Barbieri, F., & Fang, Y. (2024).
> Evaluating Very Long-Term Conversational Memory of LLM Agents.
> In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024).
> https://arxiv.org/abs/2402.17753
