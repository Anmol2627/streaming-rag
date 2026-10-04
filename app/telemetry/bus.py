"""
Streaming Live RAG — Telemetry Bus
Append-only structured JSON-lines event log. Every pipeline stage writes here.
Supports real-time SSE streaming to the dashboard and G6 schema validation.
"""
from __future__ import annotations
import json
import time
import asyncio
from pathlib import Path
from collections import defaultdict
from typing import AsyncGenerator

from app.models import TelemetryEvent


class TelemetryBus:
    """Central telemetry sink — append-only, per-session, with SSE broadcast."""

    def __init__(self, log_dir: Path | None = None):
        self._log_dir = log_dir or Path("telemetry_logs")
        self._log_dir.mkdir(parents=True, exist_ok=True)
        # In-memory per-session event lists (ephemeral)
        self._events: dict[str, list[dict]] = defaultdict(list)
        # SSE subscribers per session
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)

    def emit(self, event: TelemetryEvent) -> None:
        """Write a structured telemetry event."""
        record = event.model_dump()
        record["timestamp"] = record.get("timestamp") or time.time()
        self._events[event.session_id].append(record)
        # Write to JSON-lines file
        log_file = self._log_dir / f"{event.session_id}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        # Broadcast to SSE subscribers
        for q in self._subscribers.get(event.session_id, []):
            try:
                q.put_nowait(record)
            except asyncio.QueueFull:
                pass  # Drop if subscriber is slow

    def get_events(self, session_id: str) -> list[dict]:
        """Get all events for a session (for benchmark harness)."""
        return list(self._events.get(session_id, []))

    def get_events_by_type(self, session_id: str, event_type: str) -> list[dict]:
        """Get events of a specific type for a session."""
        return [e for e in self._events.get(session_id, [])
                if e.get("event") == event_type]

    def subscribe(self, session_id: str) -> asyncio.Queue:
        """Subscribe to real-time events for a session (for SSE)."""
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._subscribers[session_id].append(q)
        return q

    def unsubscribe(self, session_id: str, q: asyncio.Queue) -> None:
        """Remove a subscriber."""
        subs = self._subscribers.get(session_id, [])
        if q in subs:
            subs.remove(q)

    async def stream_events(self, session_id: str) -> AsyncGenerator[dict, None]:
        """Async generator for SSE streaming."""
        q = self.subscribe(session_id)
        try:
            while True:
                event = await q.get()
                yield event
        finally:
            self.unsubscribe(session_id, q)

    def clear_session(self, session_id: str) -> None:
        """Clear telemetry for a session (session teardown)."""
        self._events.pop(session_id, None)
        self._subscribers.pop(session_id, None)

    def validate_schema_completeness(self, session_id: str, turn_id: str) -> dict:
        """
        G6 gate check: verify all required event types exist for a turn.
        Returns {complete: bool, missing: [str], present: [str]}.
        """
        required_events = {
            "controller_decision", "turn_summary"
        }
        # If a RETRIEVE happened, these are also required
        turn_events = [
            e for e in self._events.get(session_id, [])
            if e.get("turn_id") == turn_id
        ]
        event_types = {e["event"] for e in turn_events}

        if "retrieval_started" in event_types or any(
            e.get("decision") == "RETRIEVE"
            for e in turn_events
            if e.get("event") == "controller_decision"
        ):
            required_events.update({
                "retrieval_started", "retrieval_completed",
                "decomposition", "fusion", "claim_generated"
            })

        missing = required_events - event_types
        return {
            "complete": len(missing) == 0,
            "missing": sorted(missing),
            "present": sorted(event_types),
        }


# Singleton instance
telemetry_bus = TelemetryBus()
