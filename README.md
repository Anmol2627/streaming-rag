# Streaming Live RAG

## PRISM_GENAI_HACKATHON_Y2026 Submission Details

**TAG**: `PRISM_GENAI_HACKATHON_Y2026`

### Submission Checklist
- [x] **Source Code**: Available in this repository.
- [x] **Presentation**: [Link to Presentation (Google Slides)](https://docs.google.com/presentation/d/1oy2t1gb010c9cifFBJCETGIDLeRl_oOfcNjDpmElZSY/edit?usp=sharing)
- [x] **Video**: [Demo Video Link (Google Drive)](https://drive.google.com/file/d/1vGF9UDgMhknset2pSfAOgK-sewjV5LZr/view?usp=sharing)
- [x] **AI Disclosure**: Included in repository as [LangAI3.0_AI_Disclosure_Filled.docx](https://github.com/Anmol2627/streaming-rag/blob/main/LangAI3.0_AI_Disclosure_Filled.docx)
- [x] **README**: Detailed ReadMe file included (this file).
- [ ] **APK/SDK (if any)**: Not applicable.
- [x] Added `requirements.txt` to the Github project.

*Refer FAQ for more details on submission guidelines.*

---

A stateful, event-driven RAG pipeline built for Samsung PRISM Theme 04. It continuously evaluates a live audio transcript stream — triggering corpus retrieval **before** the speaker finishes their sentence, decomposing compound questions into parallel sub-queries, and maintaining a versioned answer that updates as new details arrive mid-session.

---

## Quick Start

### Option A: Docker

```bash
cp .env.example .env
# Add your Groq API key to .env (free at https://console.groq.com)
docker compose up --build
```

Open **http://localhost:8000** in your browser.

> First startup takes ~30 seconds to download the MiniLM embedding model and build the FAISS index. Subsequent starts use the cached index from `.index_cache/`.

### Option B: Python (no Docker)

```bash
# Python 3.11+ required
pip install -r requirements.txt
cp .env.example .env
uvicorn app.api.server:app --host 0.0.0.0 --port 8000
```

> **No Groq key?** The system gracefully falls back to extractive synthesis. All retrieval, controller, and telemetry features remain fully operational.

---

## Testing on Your Own Private Corpus

Run the setup script with the path to your documents directory:

```bash
python scripts/setup_corpus.py path/to/your/documents
```

The script will:
- Validate that the directory contains `.txt` or `.md` files
- Update `CORPUS_PATH` in your `.env` automatically
- Clear the index cache so the server rebuilds on your corpus
- Print a per-document chunk count preview
- Tell you exactly what to run next

Then start the server as normal — the FAISS and BM25 indexes rebuild automatically (~10–30 seconds depending on corpus size).

**Document format**: Each file should be plain text (`.txt` or `.md`). Use `§` markers or double blank lines to separate sections — the chunker splits on these. The filename becomes the document ID (e.g., `policy_travel.txt` → `Doc_policy_travel`).

---

## Benchmark Results

Evaluated against 38 document-grounded scenarios from the included 48-scenario suite (`benchmark_suite.json`), using strict **Top-K = 3** retrieval to simulate real-world precision constraints.

### Retrieval Modality Comparison

| Modality | Recall @ 3 | Precision @ 3 | F1 @ 3 |
|---|---|---|---|
| Hybrid (Dense + BM25) | 0.825 | 0.649 | **0.706** |
| Dense-only (FAISS/MiniLM) | **0.877** | — | — |
| Sparse-only (BM25) | 0.833 | — | — |

Dense-only achieves higher recall at Top-K 3 on this small corpus, but Hybrid's F1 of 0.706 reflects better precision — it surfaces fewer irrelevant documents alongside the correct ones. At larger corpus scale, BM25's exact-match signal becomes critical for policy terminology where dense embeddings underperform.

### Controller Decision Accuracy

| Metric | Value |
|---|---|
| Decision accuracy | **0.704** (50/71 decisions) |
| Tier-1 decision latency | < 10ms per chunk |
| Tier-2 invocation rate | < 5% of chunks |

The Tier-1 heuristic correctly classifies 70.4% of chunk-level WAIT/RETRIEVE/NO_RETRIEVAL decisions without any LLM call. Misclassifications cluster in short late-detail fragments (e.g. `"...actually 50"`) with insufficient entity signal — exactly the cases handled by Tier-2.

### Decomposition Ablation (9 multi-intent scenarios)

| Condition | Doc-Recall @ 3 | Early Retrieval (G3) |
|---|---|---|
| Decomposition ON | **0.963** | **100% pass** |
| Decomposition OFF | 0.926 | 0% pass |

Turning decomposition off collapses early retrieval triggering entirely — the controller cannot isolate a stable sub-intent from a monolithic compound query.

### Latency (live golden scenario runs)

| Metric | Observed |
|---|---|
| Time to First Retrieval (TTFR) | ~0ms (triggers mid-stream) |
| Retrieval latency per sub-query | 27–46ms |
| Time to First Token (TTFT) | 2,332ms (MI-01), 2,635ms (ER-01) |
| Tier-1 controller decision | < 10ms (no LLM involved) |

Run `python scripts/run_benchmark_eval.py` to reproduce all figures above.

---

## Architecture

```
Transcript Chunks → [Controller] → [Decomposer] → [Hybrid Retrieval] → [Session State] → [Synthesizer]
```

**Five-stage pipeline:**

1. **Controller** — Tier-1 heuristic (entity count, semantic drift, keyword pattern) classifies each chunk as `RETRIEVE`, `WAIT`, or `NO_RETRIEVAL` in <10ms. Tier-2 LLM handles genuinely ambiguous cases.
2. **Decomposer** — LLM decomposes compound utterances into independent sub-queries. Falls back to rule-based conjunction/ellipsis splitting when the LLM is unavailable.
3. **Hybrid Retrieval** — FAISS dense search (MiniLM-L6) + BM25 sparse search per sub-query, parallelized via `ThreadPoolExecutor`. Merged with Reciprocal Rank Fusion and deduplication.
4. **Session State** — Claim-addressable versioned store. When late-arriving details change the answer, only affected claims are re-synthesized; others are preserved.
5. **Synthesizer** — Generates grounded claims from evidence. Citation validator rejects any claim citing a document not present in the retrieved evidence set.

```
app/
├── api/server.py             FastAPI + SSE endpoints
├── pipeline.py               Orchestrates all 5 stages
├── controller/controller.py  Two-tier retrieval decision engine
├── decomposer/decomposer.py  Multi-intent decomposer + rule fallback
├── retrieval/corpus.py       FAISS + BM25 index builder
├── retrieval/hybrid.py       Parallel hybrid search + RRF fusion
├── session/session_store.py  Versioned claim store
├── synthesis/synthesis.py    Grounded synthesis + citation validation
├── telemetry/bus.py          Append-only JSONL event log + SSE feed
└── config.py                 All parameters via environment variables
```

---

## Configuration

All parameters are set in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Free at console.groq.com |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Model for decomposition and synthesis |
| `CORPUS_PATH` | `streaming-rag-dev-demo-corpus/...` | Path to document directory |
| `CONTROLLER_DRIFT_THRESHOLD` | `0.15` | Semantic drift threshold for RETRIEVE trigger |
| `CONTROLLER_MAX_WAIT_CHUNKS` | `10` | Max chunks before forced retrieval |
| `RETRIEVAL_TOP_K` | `20` | Chunks retrieved per sub-query |
| `RETRIEVAL_MIN_RELEVANCE` | `0.25` | Minimum fused score for evidence inclusion |
| `MAX_SUB_QUERIES` | `5` | Max sub-queries from one decomposition |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/session` | Create a new session |
| `POST` | `/session/{id}/chunk` | Send a transcript chunk; returns controller decision |
| `POST` | `/session/{id}/turn` | End turn and synthesize answer |
| `GET` | `/session/{id}/telemetry` | SSE stream of live telemetry events |
| `GET` | `/session/{id}/telemetry/all` | Full telemetry dump for a session |
| `GET` | `/health` | Health check |
| `GET` | `/dashboard/` | Live dashboard UI |

---

## Corpus (Included)

24 documents, 153 chunks: event venues (Pune + Bangalore), catering policies, travel and expense, employee handbook, IT security, remote work, office facilities, visa procedures, meeting rooms, vendor conduct, and data privacy.

Benchmark suite: `streaming-rag-dev-demo-corpus/corpus/benchmarks/benchmark_suite.json` — 48 scenarios across 6 categories (Early Retrieval, Multi-Intent, Late Detail, Uncertainty, Deduplication, Conflict Resolution).

---

## Submission Deliverables

| Deliverable | File |
|---|---|
| Architecture Brief (G2) | `docs/ARCHITECTURE_BRIEF.md` |
| Benchmarking Report (G3) | `docs/BENCHMARK_REPORT.md` |
| Telemetry Schema (G6) | `docs/TELEMETRY_SCHEMA.md` |