"""
Streaming Live RAG — Data Models
All Pydantic models for the pipeline: chunks, controller decisions, sub-queries,
retrieved chunks, claims, answer objects, and telemetry events.
"""
from __future__ import annotations
import time
import uuid
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ─── Enums ───────────────────────────────────────────────────────────────────

class ControllerDecision(str, Enum):
    WAIT = "WAIT"
    RETRIEVE = "RETRIEVE"
    NO_RETRIEVAL = "NO_RETRIEVAL"


class TriggerType(str, Enum):
    PROVISIONAL = "provisional"
    MULTI_INTENT = "multi_intent"
    LATE_DETAIL = "late_detail"


class ClaimStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"


# ─── Turn Input ──────────────────────────────────────────────────────────────

class TranscriptChunk(BaseModel):
    """A single incoming transcript chunk with timing metadata."""
    session_id: str
    turn_id: str
    chunk_ts: float = Field(default_factory=time.time)
    chunk_text: str
    chunk_index: int = 0


# ─── Controller ──────────────────────────────────────────────────────────────

class ControllerState(BaseModel):
    """Internal state held by the Retrieval Controller across chunks."""
    buffer: str = ""
    chunk_count: int = 0
    prev_embedding: Optional[list[float]] = None
    last_decision: ControllerDecision = ControllerDecision.WAIT
    last_decision_ts: float = 0.0
    entities_seen: list[str] = Field(default_factory=list)
    has_triggered_retrieve: bool = False


class ControllerOutput(BaseModel):
    """Output of a single controller evaluation."""
    decision: ControllerDecision
    reason: str
    tier_used: int = 1  # 1 = heuristic, 2 = LLM
    trigger_type: Optional[TriggerType] = None
    chunk_ts: float = 0.0
    buffer_snapshot: str = ""


# ─── Decomposer ─────────────────────────────────────────────────────────────

class SubQuery(BaseModel):
    """A single, independently-retrievable sub-query."""
    id: str = Field(default_factory=lambda: f"sq_{uuid.uuid4().hex[:8]}")
    text: str
    entities: list[str] = Field(default_factory=list)


# ─── Retrieval ───────────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """A single retrieved corpus chunk with scoring metadata."""
    chunk_id: str
    doc_id: str
    section: str
    text: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fused_score: float = 0.0
    sub_query_id: str = ""


# ─── Claims & Answer ────────────────────────────────────────────────────────

class Claim(BaseModel):
    """A single factual claim with citation traceability."""
    claim_id: str = Field(default_factory=lambda: f"c_{uuid.uuid4().hex[:8]}")
    text: str
    citations: list[str] = Field(default_factory=list)  # ["Doc_01 §2"]
    source_sub_query: str = ""
    status: ClaimStatus = ClaimStatus.ACTIVE
    version_introduced: int = 1


class AnswerObject(BaseModel):
    """The versioned, claim-addressable answer for a session."""
    session_id: str
    version: int = 0
    claims: list[Claim] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)


# ─── Telemetry Events ───────────────────────────────────────────────────────

class TelemetryEvent(BaseModel):
    """Base telemetry event — all events share session/turn correlation."""
    event: str
    session_id: str
    turn_id: str
    timestamp: float = Field(default_factory=time.time)


class ControllerDecisionEvent(TelemetryEvent):
    event: str = "controller_decision"
    chunk_ts: float = 0.0
    decision: str = ""
    reason: str = ""
    tier_used: int = 1


class RetrievalEvent(TelemetryEvent):
    event: str = "retrieval_started"
    sub_query_id: str = ""
    mode: str = "hybrid"
    latency_ms: float = 0.0
    candidate_count: int = 0


class DecompositionEvent(TelemetryEvent):
    event: str = "decomposition"
    raw_utterance: str = ""
    sub_queries: list[dict] = Field(default_factory=list)
    merge_events: list[str] = Field(default_factory=list)


class FusionEvent(TelemetryEvent):
    event: str = "fusion"
    sub_query_id: str = ""
    dedup_count: int = 0
    final_ranked_chunk_ids: list[str] = Field(default_factory=list)


class ClaimGeneratedEvent(TelemetryEvent):
    event: str = "claim_generated"
    claim_id: str = ""
    citations: list[str] = Field(default_factory=list)
    validator_result: str = "pass"  # pass | fail | regenerated


class AnswerVersionEvent(TelemetryEvent):
    event: str = "answer_version"
    version: int = 0
    claims_changed: list[str] = Field(default_factory=list)
    claims_preserved: list[str] = Field(default_factory=list)
    trigger: str = ""


class UncertaintyEvent(TelemetryEvent):
    event: str = "uncertainty_flagged"
    sub_query_id: str = ""
    reason: str = ""


class TurnSummaryEvent(TelemetryEvent):
    event: str = "turn_summary"
    time_to_first_retrieval_ms: float = 0.0
    time_to_first_token_ms: float = 0.0
    total_tokens: int = 0
    estimated_cost_per_turn: float = 0.0
    model_used: str = ""


# ─── Session ────────────────────────────────────────────────────────────────

class Session(BaseModel):
    """Ephemeral, conversation-scoped session state."""
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    controller_state: ControllerState = Field(default_factory=ControllerState)
    answer: AnswerObject = Field(default_factory=lambda: AnswerObject(session_id=""))
    turn_count: int = 0
    created_at: float = Field(default_factory=time.time)

    def model_post_init(self, __context):
        if not self.answer.session_id:
            self.answer.session_id = self.session_id
