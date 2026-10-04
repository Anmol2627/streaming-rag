# Telemetry & Observability Schema
### Streaming Live RAG — Samsung PRISM Theme 04

---

## Overview

Every pipeline stage emits structured JSON-lines events to the **TelemetryBus** (`app/telemetry/bus.py`). Events are:
- Written append-only to `telemetry_logs/{session_id}.jsonl`
- Broadcast in real-time via SSE to the dashboard (`/session/{id}/telemetry`)
- Queryable via `/session/{id}/telemetry/all`

---

## Event Schema Reference

### 1. `controller_decision`
Emitted after each transcript chunk is evaluated.

```json
{
  "event": "controller_decision",
  "session_id": "ad8dfb5a89c54c99a84d49b2b21d3998",
  "turn_id": "turn_8a89844c",
  "timestamp": 1791006486.282,
  "chunk_ts": 1791006486.229,
  "decision": "RETRIEVE",
  "reason": "intent_stable (stability=1.00, entities=6, drift=0.057)",
  "tier_used": 1
}
```

| Field | Type | Description |
|---|---|---|
| `decision` | enum | `WAIT` \| `RETRIEVE` \| `NO_RETRIEVAL` |
| `reason` | string | Human-readable trigger rationale with numeric stability/entity/drift values |
| `tier_used` | int | `1` = heuristic, `2` = LLM-assisted |

---

### 2. `decomposition`
Emitted after the multi-intent decomposer produces sub-queries.

```json
{
  "event": "decomposition",
  "session_id": "ad8dfb5a89c54c99a84d49b2b21d3998",
  "turn_id": "turn_8a89844c",
  "timestamp": 1791006491.087,
  "raw_utterance": "I need a venue for 30 people in Pune... ...what's the cancellation policy... ...and is catering available?",
  "sub_queries": [
    {"id": "sq_6d9996cc", "text": "venue capacity 30 attendees Pune", "entities": ["venue", "30", "Pune"]},
    {"id": "sq_74a1d83d", "text": "cancellation policy venue booking", "entities": ["cancellation", "policy"]},
    {"id": "sq_bbfbe3b8", "text": "catering availability Pune venue", "entities": ["catering", "Pune"]}
  ],
  "merge_events": []
}
```

| Field | Type | Description |
|---|---|---|
| `raw_utterance` | string | Full buffer text sent to decomposer |
| `sub_queries` | array | List of `{id, text, entities}` objects |
| `merge_events` | array | Descriptions of any near-duplicate merges performed |

---

### 3. `retrieval_started`
Emitted when a sub-query begins hybrid search.

```json
{
  "event": "retrieval_started",
  "session_id": "ad8dfb5a89c54c99a84d49b2b21d3998",
  "turn_id": "turn_8a89844c",
  "timestamp": 1791006491.089,
  "sub_query_id": "sq_6d9996cc",
  "mode": "hybrid",
  "latency_ms": 0.0,
  "candidate_count": 0
}
```

---

### 4. `retrieval_completed`
Emitted after hybrid search and RRF fusion completes.

```json
{
  "event": "retrieval_completed",
  "session_id": "ad8dfb5a89c54c99a84d49b2b21d3998",
  "turn_id": "turn_8a89844c",
  "timestamp": 1791006491.116,
  "sub_query_id": "sq_6d9996cc",
  "mode": "hybrid",
  "latency_ms": 27.455,
  "candidate_count": 20
}
```

| Field | Type | Description |
|---|---|---|
| `latency_ms` | float | Wall-clock retrieval time in milliseconds |
| `candidate_count` | int | Number of chunks returned before relevance filtering |

---

### 5. `fusion`
Emitted after Reciprocal Rank Fusion produces final ranked list.

```json
{
  "event": "fusion",
  "session_id": "ad8dfb5a89c54c99a84d49b2b21d3998",
  "turn_id": "turn_8a89844c",
  "timestamp": 1791006491.118,
  "sub_query_id": "sq_6d9996cc",
  "dedup_count": 0,
  "final_ranked_chunk_ids": [
    "Doc_01_s2", "Doc_02_s2", "Doc_03_s2", "Doc_01_s8",
    "Doc_02_s7", "Doc_04_s1", "Doc_20_s4", "Doc_02_s1"
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `dedup_count` | int | Number of duplicate chunks removed across sub-queries |
| `final_ranked_chunk_ids` | array | Ordered chunk IDs after RRF (highest score first) |

---

### 6. `claim_generated`
Emitted for each claim produced by the synthesizer, including validation outcome.

```json
{
  "event": "claim_generated",
  "session_id": "ad8dfb5a89c54c99a84d49b2b21d3998",
  "turn_id": "turn_8a89844c",
  "timestamp": 1791006493.447,
  "claim_id": "c_9b564c6c",
  "citations": ["Doc_01 §2"],
  "validator_result": "pass"
}
```

| Field | Type | Description |
|---|---|---|
| `claim_id` | string | Unique claim identifier |
| `citations` | array | List of `"Doc_XX §Y"` strings cited in this claim |
| `validator_result` | enum | `pass` \| `fail` \| `regenerated` |

---

### 7. `answer_version`
Emitted when the session answer is updated (new or patched).

```json
{
  "event": "answer_version",
  "session_id": "ad8dfb5a89c54c99a84d49b2b21d3998",
  "turn_id": "turn_8a89844c",
  "timestamp": 1791006498.604,
  "version": 1,
  "claims_changed": ["c_9b564c6c", "c_5ad16401", "c_210045c1", "c_bbb77b5d"],
  "claims_preserved": [],
  "trigger": "new_answer"
}
```

| Field | Type | Description |
|---|---|---|
| `version` | int | Monotonically increasing answer version number |
| `claims_changed` | array | Claim IDs added or updated this turn |
| `claims_preserved` | array | Claim IDs retained unchanged from previous version |
| `trigger` | enum | `new_answer` \| `late_detail_patch` |

---

### 8. `turn_summary`
Emitted at the end of each pipeline turn. The primary latency and cost record.

```json
{
  "event": "turn_summary",
  "session_id": "ad8dfb5a89c54c99a84d49b2b21d3998",
  "turn_id": "turn_8a89844c",
  "timestamp": 1791006498.606,
  "time_to_first_retrieval_ms": 0.055,
  "time_to_first_token_ms": 9380.61,
  "total_tokens": 0,
  "estimated_cost_per_turn": 0.0,
  "model_used": "groq/openai/gpt-oss-120b"
}
```

| Field | Type | Description |
|---|---|---|
| `time_to_first_retrieval_ms` | float | Latency from turn start to first chunk retrieved |
| `time_to_first_token_ms` | float | End-to-end pipeline latency (TTFT) |
| `total_tokens` | int | LLM tokens consumed (0 if not tracked by provider) |
| `estimated_cost_per_turn` | float | USD cost estimate (0.0 for free-tier Groq) |
| `model_used` | string | LLM model identifier |

---

### 9. `uncertainty`
Emitted when a sub-query cannot be answered from the corpus.

```json
{
  "event": "uncertainty",
  "session_id": "dd69a7b7bcd048939d1324b15c913cbb",
  "turn_id": "turn_a1b2c3d4",
  "timestamp": 1791006521.100,
  "sub_query_id": "sq_eac33788",
  "reason": "no_relevant_evidence"
}
```

| Field | Type | Description |
|---|---|---|
| `reason` | enum | `no_relevant_evidence` \| `citation_validation_failed` |

---

## G6 Schema Completeness Gate

The `TelemetryBus.validate_schema_completeness(session_id, turn_id)` method enforces that all required events are present:

**Required for every turn:**
- `controller_decision`
- `turn_summary`

**Additionally required when RETRIEVE triggered:**
- `decomposition`
- `retrieval_started`
- `retrieval_completed`
- `fusion`
- `claim_generated`

**Returns:**
```json
{
  "complete": true,
  "missing": [],
  "present": ["answer_version", "claim_generated", "controller_decision",
              "decomposition", "fusion", "retrieval_completed",
              "retrieval_started", "turn_summary"]
}
```

---

## Telemetry Log Location

```
telemetry_logs/
└── {session_id}.jsonl     # One file per session, append-only JSON-lines
```

API access:
- **SSE stream**: `GET /session/{id}/telemetry`
- **Full dump**: `GET /session/{id}/telemetry/all`

---

*Schema version: 1.0 | Streaming Live RAG | Samsung PRISM Theme 04*
