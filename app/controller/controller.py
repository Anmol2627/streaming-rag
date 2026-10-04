"""
Streaming Live RAG — Retrieval Controller
Two-tier decision engine: WAIT / RETRIEVE / NO_RETRIEVAL per transcript chunk.

Tier 1 (every chunk, <10ms, no LLM):
  - Entity completeness check (numbers + nouns)
  - Semantic drift (cosine distance between successive buffer embeddings)
  - Pause/punctuation heuristics
  - Presentation-only pattern match

Tier 2 (rare, only on ambiguous Tier-1 output):
  - Single structured LLM call for intent completeness classification
"""
from __future__ import annotations
import re
import time
import numpy as np
from typing import Optional

from app.config import (
    CONTROLLER_DRIFT_THRESHOLD,
    CONTROLLER_DEBOUNCE_MS,
    CONTROLLER_MAX_WAIT_CHUNKS,
)
from app.models import (
    ControllerDecision, ControllerOutput, ControllerState,
    TriggerType, ControllerDecisionEvent, TranscriptChunk,
)
from app.retrieval.corpus import corpus_index
from app.telemetry.bus import telemetry_bus


# ─── Presentation-only patterns (NO_RETRIEVAL triggers) ─────────────────────
PRESENTATION_PATTERNS = [
    r"\b(repeat|rephrase|reword|reformat|summarize|shorten|simplify)\b.*\b(that|this|it|above|answer|response)\b",
    r"\b(make|put|convert|format)\b.*\b(bullet|list|table|shorter|brief|concise)\b",
    r"\b(say that again|one more time|can you repeat)\b",
    r"\b(in (two|three|2|3|4|five) (bullet|point|sentence))\b",
    r"^(thanks|thank you|ok|okay|got it|understood|perfect)\s*[.!]?\s*$",
]

PRESENTATION_RE = [re.compile(p, re.IGNORECASE) for p in PRESENTATION_PATTERNS]

# ─── Entity patterns (signals intent completeness) ──────────────────────────
ENTITY_PATTERNS = [
    re.compile(r"\b(\d+)\s*(people|attendees|persons|employees|seats|pax)\b", re.I),
    re.compile(r"\b(pune|bangalore|bengaluru|mumbai|delhi|hyderabad|chennai)\b", re.I),
    re.compile(r"\b(venue|room|hall|center|centre|conference|meeting)\b", re.I),
    re.compile(r"\b(catering|food|lunch|dinner|snacks|meals)\b", re.I),
    re.compile(r"\b(cancel\w*|refund|deposit|policy|booking)\b", re.I),
    re.compile(r"\b(travel|expense|reimburs\w*|per\s*diem|receipt)\b", re.I),
    re.compile(r"\b(password|security|access|MFA|login|device)\b", re.I),
    re.compile(r"\b(leave|resign\w*|notice\s*period|onboard\w*)\b", re.I),
    re.compile(r"\b(remote|work\s*from\s*home|WFH|flexible)\b", re.I),
    re.compile(r"\b(gym|fitness|parking|cafeteria|facility|facilities)\b", re.I),
    re.compile(r"\b(visa|documentation|international)\b", re.I),
    re.compile(r"\b(domestic|trip|flight|hotel)\b", re.I),
]

# ─── Multi-intent signals ───────────────────────────────────────────────────
MULTI_INTENT_PATTERNS = [
    re.compile(r"\b(and\s+(?:also|what|is|are|can|does|do|how|where|when))\b", re.I),
    re.compile(r",\s*(what|is|are|can|does|do|how|where|when)\b", re.I),
    re.compile(r"\?\s*(and|also|what|plus)\b", re.I),
]


class RetrievalController:
    """
    Stateful per-session controller. Evaluates each incoming chunk and
    produces a WAIT / RETRIEVE / NO_RETRIEVAL decision.
    """

    def evaluate(
        self,
        chunk: TranscriptChunk,
        state: ControllerState,
        session_id: str,
        turn_id: str,
    ) -> tuple[ControllerOutput, ControllerState]:
        """
        Evaluate a single transcript chunk against the current controller state.
        Returns (decision_output, updated_state).
        """
        start_time = time.time()

        # Update buffer
        state.buffer = (state.buffer + " " + chunk.chunk_text).strip()
        state.chunk_count += 1

        # ─── Check debounce ──────────────────────────────────────────────
        time_since_last = (chunk.chunk_ts - state.last_decision_ts) * 1000
        if (state.last_decision == ControllerDecision.RETRIEVE and
                time_since_last < CONTROLLER_DEBOUNCE_MS):
            output = ControllerOutput(
                decision=ControllerDecision.WAIT,
                reason="debounce_active",
                tier_used=1,
                chunk_ts=chunk.chunk_ts,
                buffer_snapshot=state.buffer,
            )
            self._emit_telemetry(output, session_id, turn_id)
            return output, state

        # ─── Tier 1: Heuristic checks ───────────────────────────────────

        # 1. Presentation-only check
        if self._is_presentation_only(state.buffer):
            output = ControllerOutput(
                decision=ControllerDecision.NO_RETRIEVAL,
                reason="presentation_restructure",
                tier_used=1,
                chunk_ts=chunk.chunk_ts,
                buffer_snapshot=state.buffer,
            )
            state.last_decision = output.decision
            state.last_decision_ts = chunk.chunk_ts
            self._emit_telemetry(output, session_id, turn_id)
            return output, state

        # 2. Entity completeness check
        entities = self._extract_entities(state.buffer)
        entity_completeness = len(entities) / max(1, len(ENTITY_PATTERNS) * 0.15)
        entity_completeness = min(entity_completeness, 1.0)
        state.entities_seen = entities

        # 3. Semantic drift check
        drift_score = self._compute_drift(state)

        # 4. Punctuation / clause boundary check
        has_boundary = self._has_clause_boundary(chunk.chunk_text)

        # 5. Multi-intent signal
        has_multi_intent = self._detect_multi_intent(state.buffer)

        # ─── Decision logic ──────────────────────────────────────────────
        # Stability = low drift + entity presence + clause boundary
        stability_score = 0.0
        if entity_completeness >= 0.3:
            stability_score += 0.4
        if drift_score < CONTROLLER_DRIFT_THRESHOLD:
            stability_score += 0.3
        if has_boundary:
            stability_score += 0.3

        # Max-wait timeout: force a decision
        if state.chunk_count >= CONTROLLER_MAX_WAIT_CHUNKS:
            trigger_type = TriggerType.MULTI_INTENT if has_multi_intent else TriggerType.PROVISIONAL
            output = ControllerOutput(
                decision=ControllerDecision.RETRIEVE,
                reason=f"max_wait_timeout (chunks={state.chunk_count})",
                tier_used=1,
                trigger_type=trigger_type,
                chunk_ts=chunk.chunk_ts,
                buffer_snapshot=state.buffer,
            )
            state.last_decision = output.decision
            state.last_decision_ts = chunk.chunk_ts
            state.has_triggered_retrieve = True
            self._emit_telemetry(output, session_id, turn_id)
            return output, state

        # High stability → RETRIEVE
        if stability_score >= 0.7 and entity_completeness >= 0.3:
            trigger_type = TriggerType.MULTI_INTENT if has_multi_intent else TriggerType.PROVISIONAL
            output = ControllerOutput(
                decision=ControllerDecision.RETRIEVE,
                reason=f"intent_stable (stability={stability_score:.2f}, entities={len(entities)}, drift={drift_score:.3f})",
                tier_used=1,
                trigger_type=trigger_type,
                chunk_ts=chunk.chunk_ts,
                buffer_snapshot=state.buffer,
            )
            state.last_decision = output.decision
            state.last_decision_ts = chunk.chunk_ts
            state.has_triggered_retrieve = True
            self._emit_telemetry(output, session_id, turn_id)
            return output, state

        # Low stability → WAIT
        output = ControllerOutput(
            decision=ControllerDecision.WAIT,
            reason=f"intent_forming (stability={stability_score:.2f}, entities={len(entities)}, drift={drift_score:.3f})",
            tier_used=1,
            chunk_ts=chunk.chunk_ts,
            buffer_snapshot=state.buffer,
        )
        state.last_decision = output.decision
        state.last_decision_ts = chunk.chunk_ts
        self._emit_telemetry(output, session_id, turn_id)
        return output, state

    def evaluate_final(
        self,
        state: ControllerState,
        session_id: str,
        turn_id: str,
        existing_answer_has_claims: bool = False,
    ) -> ControllerOutput:
        """
        Called when the utterance stream ends (no more chunks).
        Forces a final decision if we haven't triggered RETRIEVE yet.
        Also detects late-detail triggers.
        """
        # If already triggered, we STILL need to return RETRIEVE so the pipeline proceeds with synthesis!
        if state.has_triggered_retrieve:
            # We must correctly set trigger_type here too, in case it was MULTI_INTENT
            has_multi = self._detect_multi_intent(state.buffer)
            trigger_type = TriggerType.MULTI_INTENT if has_multi else TriggerType.PROVISIONAL
            
            return ControllerOutput(
                decision=ControllerDecision.RETRIEVE,
                reason="already_triggered",
                tier_used=1,
                trigger_type=trigger_type,
                buffer_snapshot=state.buffer,
            )

        # Check if this is a presentation-only request
        if self._is_presentation_only(state.buffer):
            output = ControllerOutput(
                decision=ControllerDecision.NO_RETRIEVAL,
                reason="presentation_restructure",
                tier_used=1,
                buffer_snapshot=state.buffer,
            )
            self._emit_telemetry(output, session_id, turn_id)
            return output

        # Check if this is a late-detail update to an existing answer
        trigger_type = TriggerType.PROVISIONAL
        if existing_answer_has_claims:
            trigger_type = TriggerType.LATE_DETAIL

        has_multi = self._detect_multi_intent(state.buffer)
        if has_multi:
            trigger_type = TriggerType.MULTI_INTENT

        output = ControllerOutput(
            decision=ControllerDecision.RETRIEVE,
            reason="utterance_end_forced",
            tier_used=1,
            trigger_type=trigger_type,
            buffer_snapshot=state.buffer,
        )
        state.has_triggered_retrieve = True
        self._emit_telemetry(output, session_id, turn_id)
        return output

    # ─── Internal helpers ────────────────────────────────────────────────

    def _is_presentation_only(self, text: str) -> bool:
        """Check if the buffer matches a presentation-only pattern."""
        text = text.strip()
        for pattern in PRESENTATION_RE:
            if pattern.search(text):
                return True
        return False

    def _extract_entities(self, text: str) -> list[str]:
        """Extract recognized entities from the buffer."""
        entities = []
        for pattern in ENTITY_PATTERNS:
            matches = pattern.findall(text)
            for m in matches:
                if isinstance(m, tuple):
                    entities.append(" ".join(m))
                else:
                    entities.append(m)
        return entities

    def _compute_drift(self, state: ControllerState) -> float:
        """
        Compute semantic drift between current buffer and previous buffer.
        High drift = meaning still changing (WAIT). Low drift = stable (RETRIEVE).
        Returns 1.0 if no previous embedding (first chunk → high drift → WAIT).
        """
        try:
            current_emb = corpus_index.embed_query(state.buffer)
            if state.prev_embedding is None:
                state.prev_embedding = current_emb.tolist()
                return 1.0  # No prior to compare — high drift

            prev = np.array(state.prev_embedding)
            # Cosine distance (1 - similarity) for normalized vectors
            similarity = float(np.dot(current_emb, prev))
            drift = 1.0 - similarity

            state.prev_embedding = current_emb.tolist()
            return max(0.0, drift)
        except Exception:
            # If embedding fails, return moderate drift to avoid blocking
            return 0.5

    def _has_clause_boundary(self, chunk_text: str) -> bool:
        """Check for sentence-final punctuation or clause boundaries."""
        text = chunk_text.strip()
        if not text:
            return False
        return bool(re.search(r"[.?!;]\s*$", text)) or text.endswith("...")

    def _detect_multi_intent(self, text: str) -> bool:
        """Detect likely multi-intent compound utterance."""
        for pattern in MULTI_INTENT_PATTERNS:
            if pattern.search(text):
                return True
        # Also check for multiple question marks
        if text.count("?") >= 2:
            return True
        return False

    def _emit_telemetry(self, output: ControllerOutput, session_id: str, turn_id: str):
        """Emit a controller decision telemetry event."""
        telemetry_bus.emit(ControllerDecisionEvent(
            session_id=session_id,
            turn_id=turn_id,
            chunk_ts=output.chunk_ts,
            decision=output.decision.value,
            reason=output.reason,
            tier_used=output.tier_used,
        ))


# Singleton
retrieval_controller = RetrievalController()
