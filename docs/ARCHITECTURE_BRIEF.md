# Streaming Live RAG — System Architecture
### Samsung PRISM Theme 04

---

## 1. System Design Rationale

Traditional RAG pipelines are **turn-complete**: they wait until a user finishes speaking, then retrieve, then answer. This is acceptable for text input but unacceptable for live speech/transcript streams where latency directly degrades the user experience.

**Streaming Live RAG** solves this with a **5-stage event-driven pipeline** that continuously evaluates each incoming transcript chunk, triggering retrieval the moment the utterance has enough semantic signal — often *before the speaker finishes their sentence*.

### Design Principles

| Principle | Implementation |
|---|---|
| **Architectural Parsimony** | Every component is justified by latency/compute budget. No over-engineering. |
| **Corpus Isolation** | Zero web scraping, zero parametric memory. All facts grounded in indexed corpus only. |
| **Claim-Addressable State** | Answers are lists of versioned claims, not monolithic strings. Only affected claims are patched on new information. |
| **Observability First** | Every pipeline event is emitted to a structured telemetry bus before any response is returned. |

---

## 2. Pipeline Architecture

```
Transcript Stream
      │
      ▼
┌─────────────────────────────────────────────────┐
│  Stage 1: Retrieval Controller                   │
│  ─────────────────────────────                   │
│  Tier 1 (every chunk, <10ms, no LLM):            │
│    • Entity completeness scoring                  │
│    • Semantic drift (cosine distance)             │
│    • Pause/punctuation heuristics                 │
│    • Presentation-only pattern guard              │
│  Tier 2 (rare, ambiguous only):                  │
│    • Single structured LLM call                  │
│                                                  │
│  Outputs: WAIT | RETRIEVE | NO_RETRIEVAL         │
└─────────────────────────────┬───────────────────┘
                              │ RETRIEVE
                              ▼
┌─────────────────────────────────────────────────┐
│  Stage 2: Multi-Intent Decomposer                │
│  ──────────────────────────────                  │
│  • LLM decomposes compound utterance             │
│  • Lazy-LLM fallback: if LLM returns ≤1          │
│    query for multi-intent input → rule-based     │
│    split on clause boundaries                    │
│  • Near-duplicate merge (cosine sim > 0.9)       │
│  • Capped at MAX_SUB_QUERIES (default: 5)        │
└─────────────────────────────┬───────────────────┘
                              │ [SubQuery, ...]
                              ▼
┌─────────────────────────────────────────────────┐
│  Stage 3: Hybrid Retrieval & Fusion              │
│  ─────────────────────────────                   │
│  Per sub-query:                                  │
│    • Dense: FAISS (MiniLM-L6-v2 embeddings)      │
│    • Sparse: BM25 (rank_bm25)                    │
│    • RRF fusion: score = Σ 1/(k + rank)          │
│    • Dedup by chunk hash                         │
│    • Filter: fused_score >= MIN_RELEVANCE        │
└─────────────────────────────┬───────────────────┘
                              │ {sq_id: [RetrievedChunk]}
                              ▼
┌─────────────────────────────────────────────────┐
│  Stage 4: Session State Manager                  │
│  ─────────────────────────────                   │
│  • Claim-addressable session store               │
│  • Compares new claims to existing by embedding  │
│  • Patches only changed claims (late detail)     │
│  • Preserves unchanged claims (version stable)   │
└─────────────────────────────┬───────────────────┘
                              │ delta claims
                              ▼
┌─────────────────────────────────────────────────┐
│  Stage 5: Grounded Claim Synthesizer             │
│  ────────────────────────────────                │
│  • LLM generates claims per sub-query            │
│  • Citation validator: rejects hallucinated IDs  │
│  • Extractive fallback if LLM unavailable        │
│  • Uncertainty propagation for missing evidence  │
└─────────────────────────────────────────────────┘
```

---

## 3. Retrieval Trigger Logic

The controller is the most latency-critical component. It must decide per-chunk (every ~0.5–1s) whether to trigger retrieval — without calling an LLM for every chunk.

### Stability Score Computation

```
stability_score = 0.0
if entity_completeness >= 0.3:   stability_score += 0.4
if drift < DRIFT_THRESHOLD:      stability_score += 0.3
if has_clause_boundary:          stability_score += 0.3

if stability_score >= 0.7:  → RETRIEVE
if chunk_count >= MAX_WAIT: → RETRIEVE (timeout)
else:                       → WAIT
```

### Entity Completeness
Entities detected via 12 domain regex patterns (venues, cities, headcounts, policies, facilities). `completeness = min(matched_count / 4, 1.0)`.

### Semantic Drift
`drift = 1 - cosine_similarity(current_buffer_embedding, prev_buffer_embedding)`. High drift → meaning still changing → WAIT. Low drift → intent stabilized → candidate for RETRIEVE.

### Presentation-Only Guard
5 regex patterns detect reformatting requests ("summarize", "bullet points", "repeat"). If matched → NO_RETRIEVAL; existing answer is reformatted without corpus access.

---

## 4. Query Decomposition Strategy

### LLM Path (Multi-Intent Detected)

Multi-intent signals: `and is/are/can/does`, `, what/is`, `?...and`. On detection, the LLM decomposes:

```
Input:  "I need a venue for 30 people in Pune...what's the cancellation 
         policy...and is catering available?"

Output: [
  {"text": "venue capacity 30 attendees Pune",    "entities": ["venue","30","Pune"]},
  {"text": "cancellation policy venue booking",   "entities": ["cancellation","policy"]},
  {"text": "catering availability Pune venue",    "entities": ["catering","Pune"]}
]
```

### Rule-Based Fallback
If the LLM returns ≤1 query for a detected multi-intent utterance, a rule-based fallback splits on ellipsis patterns (`...`), coordinating conjunctions, and sentence boundaries. Ensures sub-query count ≥ 2 for compound queries.

### Merge Pass
After decomposition, sub-queries with cosine similarity > 0.9 are merged (keeping longer text). Prevents redundant retrieval for near-duplicate fragments.

---

## 5. Data Provenance & Corpus Isolation

All factual claims are grounded exclusively in the 24-document corpus (153 indexed chunks).

1. **Citation ID Validation**: Every claim's cited `Doc_XX §Y` must correspond to a real retrieved chunk. Claims citing non-retrieved IDs are rejected as hallucinated.
2. **No Parametric Memory**: Synthesis prompt explicitly instructs the LLM to use only provided evidence chunks.
3. **Uncertainty Propagation**: When no evidence passes relevance threshold, `UncertaintyEvent` is emitted and the UI displays `⚠ Insufficient Evidence`.
4. **Corpus Version Pinning**: `corpus_version` field in `benchmark_suite.json` ensures evaluation reproducibility.

---

## 6. Trade-offs & Failure Modes

### Architectural Trade-offs

| Decision | Upside | Cost |
|---|---|---|
| **Tier-1 heuristic controller** | <10ms per chunk, no API cost | Lower accuracy than LLM-based |
| **BM25 + FAISS hybrid** | Covers both exact-match and semantic | RRF fusion adds one pass overhead |
| **MiniLM-L6-v2** | CPU-viable, fast | Lower accuracy than large models |
| **Groq LLM (free tier)** | Zero cost | Rate-limited, occasional JSON errors |
| **Per-sub-query synthesis** | Bounded latency | May miss cross-intent correlations |

### Failure Modes & Mitigations

| Failure Mode | Detection | Mitigation |
|---|---|---|
| LLM returns invalid JSON | `json.JSONDecodeError` | Regex extraction fallback + trailing-comma fix |
| LLM returns 1 query for multi-intent | `len(queries) <= 1 and multi_intent_detected` | Force rule-based decomposition |
| Hallucinated citation IDs | Citation ID not in evidence set | `_validate_citations()` rejects the claim |
| Corpus has no relevant answer | All `fused_score < MIN_RELEVANCE` | Emit `UncertaintyEvent`, display amber warning |
| RETRIEVE triggered too early | Drift still high | Require `stability_score >= 0.7` |
| Redundant retrieval on reformats | Presentation pattern match | `NO_RETRIEVAL` + reformulate from existing answer |

---
