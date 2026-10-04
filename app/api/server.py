"""
Streaming Live RAG — FastAPI Server
REST + SSE API for the streaming pipeline, session management, and telemetry.
"""
from __future__ import annotations
import json
import time
import uuid
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.config import HOST, PORT, CORPUS_PATH
from app.models import TranscriptChunk, ControllerDecision
from app.pipeline import pipeline, PipelineResult
from app.retrieval.corpus import corpus_index
from app.session.session_store import session_store
from app.telemetry.bus import telemetry_bus


# ─── Lifespan: build corpus index on startup ────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build corpus index on startup."""
    print("=" * 60)
    print("Streaming Live RAG — Starting up")
    print("=" * 60)
    corpus_index.build()
    print(f"[Startup] Corpus indexed: {len(corpus_index.chunks)} chunks")
    print("=" * 60)
    yield
    print("[Shutdown] Cleaning up...")


app = FastAPI(
    title="Streaming Live RAG",
    description="Samsung PRISM Theme 04 — Streaming retrieval-augmented generation with per-chunk control, multi-intent decomposition, claim-level versioning, and full telemetry.",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve dashboard static files
dashboard_path = Path(__file__).resolve().parent.parent.parent / "dashboard"
if dashboard_path.exists():
    app.mount("/dashboard", StaticFiles(directory=str(dashboard_path), html=True), name="dashboard")


# ─── Request/Response Models ────────────────────────────────────────────────

class CreateSessionResponse(BaseModel):
    session_id: str


class ChunkRequest(BaseModel):
    chunk_text: str
    chunk_ts: float | None = None
    chunk_index: int = 0


class TurnRequest(BaseModel):
    """Complete turn: a list of transcript chunks to process."""
    chunks: list[ChunkRequest]


class AnswerResponse(BaseModel):
    session_id: str
    version: int
    claims: list[dict]
    uncertainty: list[str]
    is_no_retrieval: bool = False
    reformulated_text: str = ""
    controller_decisions: list[dict] = []
    sub_queries: list[dict] = []
    evidence_summary: list[dict] = []
    time_to_first_retrieval_ms: float = 0.0
    time_to_first_token_ms: float = 0.0


# ─── API Routes ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and load balancers."""
    return {
        "status": "healthy",
        "corpus_chunks": len(corpus_index.chunks),
        "version": "1.0.0",
    }


@app.post("/session", response_model=CreateSessionResponse)
async def create_session():
    """Create a new ephemeral session."""
    session = session_store.create_session()
    return CreateSessionResponse(session_id=session.session_id)


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    session_store.delete_session(session_id)
    telemetry_bus.clear_session(session_id)
    return {"status": "deleted"}


@app.get("/session/{session_id}/answer")
async def get_answer(session_id: str):
    """Get the current versioned answer for a session."""
    answer = session_store.get_answer(session_id)
    if not answer:
        raise HTTPException(404, "Session not found or no answer yet")
    return answer.model_dump()


@app.post("/session/{session_id}/turn", response_model=AnswerResponse)
async def process_turn(session_id: str, request: TurnRequest):
    """
    Process a complete turn (sequence of transcript chunks).
    This is the main pipeline entry point.
    """
    # Validate session
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    # Build TranscriptChunk objects
    chunks = []
    base_ts = time.time()
    for i, cr in enumerate(request.chunks):
        chunks.append(TranscriptChunk(
            session_id=session_id,
            turn_id="",  # Will be set by pipeline
            chunk_ts=cr.chunk_ts or (base_ts + i * 0.8),
            chunk_text=cr.chunk_text,
            chunk_index=cr.chunk_index or i,
        ))

    # Run pipeline
    result = pipeline.process_turn(session_id, chunks)

    # Build response
    answer = result.answer
    resp = AnswerResponse(
        session_id=session_id,
        version=answer.version if answer else 0,
        claims=[c.model_dump() for c in (answer.claims if answer else [])],
        uncertainty=answer.uncertainty if answer else [],
        is_no_retrieval=result.is_no_retrieval,
        reformulated_text=result.reformulated_text,
        controller_decisions=[
            {"decision": d.decision.value, "reason": d.reason, "tier_used": d.tier_used}
            for d in result.controller_decisions
        ],
        sub_queries=[
            {"id": sq.id, "text": sq.text, "entities": sq.entities}
            for sq in result.sub_queries
        ],
        evidence_summary=[
            {
                "sub_query_id": sq_id,
                "chunk_count": len(chunks_list),
                "top_chunks": [
                    {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "section": c.section,
                     "fused_score": round(c.fused_score, 4),
                     "text_preview": c.text[:200]}
                    for c in chunks_list[:5]
                ]
            }
            for sq_id, chunks_list in result.evidence.items()
        ],
        time_to_first_retrieval_ms=round(result.time_to_first_retrieval_ms, 2),
        time_to_first_token_ms=round(result.time_to_first_token_ms, 2),
    )
    return resp


@app.post("/session/{session_id}/chunk")
async def process_single_chunk(session_id: str, request: ChunkRequest):
    """
    Feed a single transcript chunk. Returns the controller decision.
    For streaming mode — call this per chunk, then call /turn for synthesis.
    """
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    turn_id = f"turn_{uuid.uuid4().hex[:8]}"
    chunk = TranscriptChunk(
        session_id=session_id,
        turn_id=turn_id,
        chunk_ts=request.chunk_ts or time.time(),
        chunk_text=request.chunk_text,
        chunk_index=request.chunk_index,
    )

    from app.controller.controller import retrieval_controller
    output, session.controller_state = retrieval_controller.evaluate(
        chunk, session.controller_state, session_id, turn_id
    )

    return {
        "decision": output.decision.value,
        "reason": output.reason,
        "tier_used": output.tier_used,
        "trigger_type": output.trigger_type.value if output.trigger_type else None,
        "buffer": output.buffer_snapshot,
    }


# ─── Telemetry SSE ──────────────────────────────────────────────────────────

@app.get("/session/{session_id}/telemetry")
async def telemetry_stream(session_id: str):
    """SSE stream of telemetry events for the dashboard."""
    async def event_generator():
        # First, send all existing events
        existing = telemetry_bus.get_events(session_id)
        for event in existing:
            yield {"data": json.dumps(event)}

        # Then stream new events
        q = telemetry_bus.subscribe(session_id)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield {"data": json.dumps(event)}
                except asyncio.TimeoutError:
                    yield {"data": json.dumps({"event": "heartbeat"})}
        finally:
            telemetry_bus.unsubscribe(session_id, q)

    return EventSourceResponse(event_generator())


@app.get("/session/{session_id}/telemetry/all")
async def get_all_telemetry(session_id: str):
    """Get all telemetry events for a session (for benchmark harness)."""
    events = telemetry_bus.get_events(session_id)
    return {"session_id": session_id, "events": events}


# ─── Benchmark Replay ───────────────────────────────────────────────────────

@app.post("/benchmark/replay")
async def benchmark_replay():
    """
    Run benchmark suite scenarios and compute gate metrics.
    Loads scenarios from the benchmark file and runs each through the pipeline.
    """
    benchmark_path = CORPUS_PATH.parent / "benchmarks" / "benchmark_suite.json"
    if not benchmark_path.exists():
        raise HTTPException(404, "Benchmark suite not found")

    import json as json_mod
    suite = json_mod.loads(benchmark_path.read_text(encoding="utf-8"))
    scenarios = suite.get("scenarios", [])

    results = []
    for scenario in scenarios[:5]:  # Run first 5 for quick check
        # Create a session for this scenario
        session = session_store.create_session()
        sid = session.session_id

        # Build chunks
        chunks = [
            TranscriptChunk(
                session_id=sid,
                turn_id="",
                chunk_ts=time.time() + i * 0.8,
                chunk_text=text,
                chunk_index=i,
            )
            for i, text in enumerate(scenario.get("transcript_chunks", []))
        ]

        # Process
        result = pipeline.process_turn(sid, chunks)

        # Evaluate
        expected_decisions = scenario.get("expected_controller_decisions", [])
        actual_decisions = [d.decision.value for d in result.controller_decisions]

        results.append({
            "scenario_id": scenario.get("scenario_id"),
            "category": scenario.get("category"),
            "expected_decisions": expected_decisions,
            "actual_decisions": actual_decisions[:len(expected_decisions)],
            "sub_queries_found": len(result.sub_queries),
            "expected_sub_intents": len(scenario.get("expected_sub_intents", [])),
            "answer_version": result.answer.version if result.answer else 0,
            "claims_count": len(result.answer.claims) if result.answer else 0,
        })

        # Cleanup
        session_store.delete_session(sid)

    return {"results": results}


# ─── Corpus Info ─────────────────────────────────────────────────────────────

@app.get("/corpus/info")
async def corpus_info():
    """Get corpus statistics."""
    return {
        "total_chunks": len(corpus_index.chunks),
        "documents": list(set(c.doc_id for c in corpus_index.chunks)),
        "sections_per_doc": {
            doc_id: sum(1 for c in corpus_index.chunks if c.doc_id == doc_id)
            for doc_id in set(c.doc_id for c in corpus_index.chunks)
        },
    }


# ─── Dashboard redirect ─────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to dashboard."""
    return """
    <html>
    <head><meta http-equiv="refresh" content="0; url=/dashboard/"></head>
    <body><p>Redirecting to <a href="/dashboard/">dashboard</a>...</p></body>
    </html>
    """
