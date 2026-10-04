"""
Streaming Live RAG — Pipeline Orchestrator
Coordinates the five-stage pipeline: Controller → Decomposer → Retrieval/Fusion
→ Session State → Synthesis. Event-driven, not batch. Each stage emits telemetry.
"""
from __future__ import annotations
import time
import uuid
from typing import Optional

from app.models import (
    TranscriptChunk, ControllerDecision, ControllerState, ControllerOutput,
    TriggerType, SubQuery, RetrievedChunk, Claim, AnswerObject, Session,
    TurnSummaryEvent,
)
from app.controller.controller import retrieval_controller
from app.decomposer.decomposer import decompose_utterance
from app.retrieval.hybrid import retrieve_and_fuse
from app.session.session_store import session_store
from app.synthesis.synthesis import synthesize_claims, reformulate_answer
from app.telemetry.bus import telemetry_bus


class PipelineResult:
    """Result of processing a complete turn through the pipeline."""
    def __init__(self):
        self.session_id: str = ""
        self.turn_id: str = ""
        self.controller_decisions: list[ControllerOutput] = []
        self.sub_queries: list[SubQuery] = []
        self.evidence: dict[str, list[RetrievedChunk]] = {}
        self.answer: AnswerObject | None = None
        self.is_no_retrieval: bool = False
        self.reformulated_text: str = ""
        self.time_to_first_retrieval_ms: float = 0.0
        self.time_to_first_token_ms: float = 0.0
        self.total_tokens: int = 0


class Pipeline:
    """
    Main pipeline orchestrator. Processes transcript chunks through the
    five-stage pipeline.
    """

    def process_turn(
        self,
        session_id: str,
        chunks: list[TranscriptChunk],
    ) -> PipelineResult:
        """
        Process a complete turn (sequence of transcript chunks) through the pipeline.
        This is the main entry point for the streaming pipeline.
        """
        turn_start = time.time()
        turn_id = f"turn_{uuid.uuid4().hex[:8]}"
        result = PipelineResult()
        result.session_id = session_id
        result.turn_id = turn_id

        # Get or create session
        session = session_store.get_session(session_id)
        if not session:
            session = session_store.create_session()
            session_id = session.session_id
            result.session_id = session_id

        # Use existing controller state
        state = session.controller_state
        
        # If the backend is processing a fresh batch of chunks (not streaming), we might want to reset,
        # but for now we just accumulate.
        retrieval_triggered = False
        trigger_output: ControllerOutput | None = None
        first_retrieval_ts: float | None = None

        # ─── Stage 1: Retrieval Controller ───────────────────────────
        for chunk in chunks:
            chunk.session_id = session_id
            chunk.turn_id = turn_id

            output, state = retrieval_controller.evaluate(
                chunk, state, session_id, turn_id
            )
            result.controller_decisions.append(output)

            if output.decision == ControllerDecision.RETRIEVE and not retrieval_triggered:
                retrieval_triggered = True
                trigger_output = output
                first_retrieval_ts = time.time()

            if output.decision == ControllerDecision.NO_RETRIEVAL:
                # Short-circuit: no retrieval needed
                result.is_no_retrieval = True
                break

        # If no RETRIEVE triggered during streaming, force final decision
        if not retrieval_triggered and not result.is_no_retrieval:
            existing_has_claims = bool(session.answer.claims)
            final_output = retrieval_controller.evaluate_final(
                state, session_id, turn_id,
                existing_answer_has_claims=existing_has_claims,
            )
            result.controller_decisions.append(final_output)

            if final_output.decision == ControllerDecision.RETRIEVE:
                retrieval_triggered = True
                trigger_output = final_output
                first_retrieval_ts = time.time()
            elif final_output.decision == ControllerDecision.NO_RETRIEVAL:
                result.is_no_retrieval = True

        # ─── Handle NO_RETRIEVAL (presentation-only) ─────────────────
        if result.is_no_retrieval:
            current_answer_text = self._answer_to_text(session.answer)
            if current_answer_text:
                full_utterance = state.buffer
                reformulated = reformulate_answer(
                    current_answer_text, full_utterance, session_id, turn_id
                )
                result.reformulated_text = reformulated
            result.answer = session.answer
            self._emit_turn_summary(result, turn_start, session_id, turn_id)
            return result

        if not retrieval_triggered or not trigger_output:
            # Nothing to do
            result.answer = session.answer
            self._emit_turn_summary(result, turn_start, session_id, turn_id)
            return result

        # ─── Stage 2: Multi-Intent Decomposition ────────────────────
        full_utterance = state.buffer
        trigger_type = trigger_output.trigger_type or TriggerType.PROVISIONAL

        sub_queries = decompose_utterance(
            full_utterance, session_id, turn_id, trigger_type
        )
        result.sub_queries = sub_queries

        # ─── Stage 3: Hybrid Retrieval & Fusion ─────────────────────
        evidence = retrieve_and_fuse(sub_queries, session_id, turn_id)
        result.evidence = evidence

        if first_retrieval_ts:
            result.time_to_first_retrieval_ms = (first_retrieval_ts - turn_start) * 1000

        # ─── Stage 4 & 5: Session State + Synthesis ──────────────────
        if trigger_type == TriggerType.LATE_DETAIL:
            # Late-detail: targeted patch
            result.answer = self._process_late_detail(
                session, full_utterance, sub_queries, evidence,
                session_id, turn_id
            )
        else:
            # Fresh answer
            result.answer = self._process_fresh_answer(
                session, sub_queries, evidence, session_id, turn_id
            )

        result.time_to_first_token_ms = (time.time() - turn_start) * 1000
        self._emit_turn_summary(result, turn_start, session_id, turn_id)
        return result

    def _process_fresh_answer(
        self,
        session: Session,
        sub_queries: list[SubQuery],
        evidence: dict[str, list[RetrievedChunk]],
        session_id: str,
        turn_id: str,
    ) -> AnswerObject:
        """Generate a fresh answer from scratch."""
        all_claims = []
        all_uncertainty = []

        for sq in sub_queries:
            sq_evidence = evidence.get(sq.id, [])
            claims, uncertainties = synthesize_claims(
                sq, sq_evidence, session_id, turn_id
            )
            all_claims.extend(claims)
            all_uncertainty.extend(uncertainties)

        return session_store.update_answer(
            session_id=session_id,
            new_claims=all_claims,
            uncertainty=all_uncertainty,
            turn_id=turn_id,
            is_late_detail=False,
        )

    def _process_late_detail(
        self,
        session: Session,
        utterance: str,
        sub_queries: list[SubQuery],
        evidence: dict[str, list[RetrievedChunk]],
        session_id: str,
        turn_id: str,
    ) -> AnswerObject:
        """Process a late-arriving detail: patch only affected claims."""
        # Flatten all evidence
        all_evidence = []
        for sq_evidence in evidence.values():
            all_evidence.extend(sq_evidence)

        # Detect affected claims
        affected_ids = session_store.detect_affected_claims(
            session_id, utterance, all_evidence
        )

        if not affected_ids:
            # No claims affected — treat as fresh additional claims
            return self._process_fresh_answer(
                session, sub_queries, evidence, session_id, turn_id
            )

        # Generate replacement claims only for affected sub-intents
        new_claims = []
        new_uncertainty = []

        for sq in sub_queries:
            sq_evidence = evidence.get(sq.id, [])
            claims, uncertainties = synthesize_claims(
                sq, sq_evidence, session_id, turn_id
            )
            new_claims.extend(claims)
            new_uncertainty.extend(uncertainties)

        return session_store.update_answer(
            session_id=session_id,
            new_claims=new_claims,
            uncertainty=new_uncertainty,
            turn_id=turn_id,
            is_late_detail=True,
            affected_claim_ids=affected_ids,
        )

    def _answer_to_text(self, answer: AnswerObject) -> str:
        """Convert an answer object to plain text for reformulation."""
        if not answer.claims:
            return ""
        parts = []
        for claim in answer.claims:
            parts.append(claim.text)
        if answer.uncertainty:
            parts.append("\n".join(answer.uncertainty))
        return "\n".join(parts)

    def _emit_turn_summary(
        self,
        result: PipelineResult,
        turn_start: float,
        session_id: str,
        turn_id: str,
    ):
        """Emit turn summary telemetry."""
        telemetry_bus.emit(TurnSummaryEvent(
            session_id=session_id,
            turn_id=turn_id,
            time_to_first_retrieval_ms=result.time_to_first_retrieval_ms,
            time_to_first_token_ms=result.time_to_first_token_ms,
            total_tokens=result.total_tokens,
            estimated_cost_per_turn=0.0,  # Groq free tier
            model_used="groq/" + (LLM_MODEL if GROQ_API_KEY else "extractive_fallback"),
        ))


# Import here to avoid circular imports
from app.config import LLM_MODEL, GROQ_API_KEY

# Singleton
pipeline = Pipeline()
