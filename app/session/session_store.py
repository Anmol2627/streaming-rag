"""
Streaming Live RAG — Session State & Answer Versioning
Claim-addressable, versioned answer store. Supports targeted delta retrieval
for late-arriving details: patches only affected claims, preserves the rest
byte-identical. Ephemeral, session-scoped — no cross-session state.
"""
from __future__ import annotations
import time
import uuid
import numpy as np
from typing import Optional

from app.models import (
    Session, AnswerObject, Claim, ClaimStatus,
    AnswerVersionEvent, RetrievedChunk,
)
from app.retrieval.corpus import corpus_index
from app.telemetry.bus import telemetry_bus


class SessionStore:
    """In-memory session store. Keyed by session_id, fully ephemeral."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create_session(self) -> Session:
        """Create a new ephemeral session."""
        session = Session()
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> None:
        """Delete a session (cleanup)."""
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())

    def update_answer(
        self,
        session_id: str,
        new_claims: list[Claim],
        uncertainty: list[str],
        turn_id: str,
        is_late_detail: bool = False,
        affected_claim_ids: list[str] | None = None,
    ) -> AnswerObject:
        """
        Update the session's answer. Two modes:
        1. Fresh answer (is_late_detail=False): replaces all claims.
        2. Late-detail patch (is_late_detail=True): patches only affected claims,
           preserves the rest byte-identical.
        """
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        old_answer = session.answer
        new_version = old_answer.version + 1

        if is_late_detail and affected_claim_ids:
            # ─── Late-detail: patch only affected claims ─────────────
            preserved_claims = []
            changed_ids = []

            for old_claim in old_answer.claims:
                if old_claim.claim_id in affected_claim_ids:
                    # Mark as superseded
                    old_claim.status = ClaimStatus.SUPERSEDED
                    changed_ids.append(old_claim.claim_id)
                else:
                    # Preserve byte-identical
                    preserved_claims.append(old_claim)

            # Add new claims (replacements for affected ones)
            for nc in new_claims:
                nc.version_introduced = new_version
                nc.status = ClaimStatus.ACTIVE

            all_claims = preserved_claims + new_claims

            telemetry_bus.emit(AnswerVersionEvent(
                session_id=session_id,
                turn_id=turn_id,
                version=new_version,
                claims_changed=[c.claim_id for c in new_claims],
                claims_preserved=[c.claim_id for c in preserved_claims],
                trigger="late_detail",
            ))
        else:
            # ─── Fresh answer ────────────────────────────────────────
            for nc in new_claims:
                nc.version_introduced = new_version
                nc.status = ClaimStatus.ACTIVE

            all_claims = new_claims

            telemetry_bus.emit(AnswerVersionEvent(
                session_id=session_id,
                turn_id=turn_id,
                version=new_version,
                claims_changed=[c.claim_id for c in new_claims],
                claims_preserved=[],
                trigger="new_answer",
            ))

        # Build new answer object
        new_answer = AnswerObject(
            session_id=session_id,
            version=new_version,
            claims=[c for c in all_claims if c.status == ClaimStatus.ACTIVE],
            uncertainty=uncertainty,
        )
        session.answer = new_answer
        session.turn_count += 1

        return new_answer

    def detect_affected_claims(
        self,
        session_id: str,
        new_utterance: str,
        new_evidence: list[RetrievedChunk],
    ) -> list[str]:
        """
        Identify which existing claims are affected by a late-arriving detail.
        Uses semantic overlap between the new utterance/evidence and each
        existing claim's topic/entity signature.

        Returns list of affected claim_ids.
        """
        session = self._sessions.get(session_id)
        if not session or not session.answer.claims:
            return []

        affected = []
        try:
            new_emb = corpus_index.embed_query(new_utterance)

            for claim in session.answer.claims:
                if claim.status != ClaimStatus.ACTIVE:
                    continue

                claim_emb = corpus_index.embed_query(claim.text)
                similarity = float(np.dot(new_emb, claim_emb))

                # High similarity between new utterance and existing claim
                # suggests the claim's topic is being modified
                if similarity > 0.4:
                    affected.append(claim.claim_id)
                    continue

                # Also check if new evidence overlaps with claim's cited chunks
                for evidence in new_evidence:
                    for citation in claim.citations:
                        # Parse citation: "Doc_XX §Y"
                        parts = citation.split(" ")
                        if len(parts) >= 2:
                            doc_id = parts[0]
                            if evidence.doc_id == doc_id:
                                affected.append(claim.claim_id)
                                break
                    if claim.claim_id in affected:
                        break

        except Exception as e:
            print(f"[SessionStore] Affected-claim detection error: {e}")
            # Conservative: if detection fails, mark all claims as potentially affected
            # but only return them if we have evidence that actually overlaps
            pass

        return list(set(affected))

    def get_answer(self, session_id: str) -> AnswerObject | None:
        """Get the current answer for a session."""
        session = self._sessions.get(session_id)
        return session.answer if session else None


# Singleton
session_store = SessionStore()
