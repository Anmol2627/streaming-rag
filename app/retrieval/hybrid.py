"""
Streaming Live RAG — Hybrid Retrieval & Fusion
Per sub-query: dense + sparse search → Reciprocal Rank Fusion → dedup → rerank.
"""
from __future__ import annotations
import time
import hashlib
from collections import defaultdict

from app.config import RETRIEVAL_TOP_K, RETRIEVAL_FUSION_K, RETRIEVAL_MIN_RELEVANCE
from app.models import RetrievedChunk, SubQuery
from app.models import RetrievalEvent, FusionEvent
from app.retrieval.corpus import corpus_index, CorpusIndex
from app.telemetry.bus import telemetry_bus


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60
) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion: combines multiple ranked lists into one.
    score(doc) = Σ 1/(k + rank_in_list) across all lists.
    """
    scores: dict[int, float] = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, (idx, _original_score) in enumerate(ranked_list):
            scores[idx] += 1.0 / (k + rank + 1)  # rank is 0-based
    # Sort by fused score descending
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def deduplicate_chunks(
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """
    Deduplicate retrieved chunks by chunk_id and content hash.
    Retains the highest fused_score for duplicate entries.
    """
    seen_ids: dict[str, RetrievedChunk] = {}
    seen_hashes: dict[str, RetrievedChunk] = {}

    for chunk in chunks:
        content_hash = hashlib.md5(chunk.text.encode()).hexdigest()

        # Dedup by chunk_id
        if chunk.chunk_id in seen_ids:
            existing = seen_ids[chunk.chunk_id]
            if chunk.fused_score > existing.fused_score:
                seen_ids[chunk.chunk_id] = chunk
            continue

        # Dedup by content hash (near-duplicate)
        if content_hash in seen_hashes:
            existing = seen_hashes[content_hash]
            if chunk.fused_score > existing.fused_score:
                # Replace with higher-scoring version
                seen_ids.pop(existing.chunk_id, None)
                seen_ids[chunk.chunk_id] = chunk
                seen_hashes[content_hash] = chunk
            continue

        seen_ids[chunk.chunk_id] = chunk
        seen_hashes[content_hash] = chunk

    return list(seen_ids.values())


def hybrid_retrieve(
    sub_query: SubQuery,
    session_id: str,
    turn_id: str,
    top_k: int | None = None,
    index: CorpusIndex | None = None,
) -> list[RetrievedChunk]:
    """
    Hybrid retrieval for a single sub-query:
    1. Dense search (embedding cosine similarity)
    2. Sparse search (BM25)
    3. RRF fusion
    4. Convert to RetrievedChunk objects with scoring metadata
    """
    top_k = top_k or RETRIEVAL_TOP_K
    idx = index or corpus_index
    start_time = time.time()

    # Emit retrieval_started
    telemetry_bus.emit(RetrievalEvent(
        event="retrieval_started",
        session_id=session_id,
        turn_id=turn_id,
        sub_query_id=sub_query.id,
        mode="hybrid",
    ))

    # Dense search
    dense_results = idx.dense_search(sub_query.text, top_k=top_k)

    # Sparse search
    sparse_results = idx.sparse_search(sub_query.text, top_k=top_k)

    # RRF fusion
    fused = reciprocal_rank_fusion(
        [dense_results, sparse_results],
        k=RETRIEVAL_FUSION_K
    )

    # Convert to RetrievedChunk objects
    chunks = []
    # Build score lookup maps
    dense_map = {idx_val: score for idx_val, score in dense_results}
    sparse_map = {idx_val: score for idx_val, score in sparse_results}

    for chunk_idx, fused_score in fused[:top_k]:
        corpus_chunk = idx.get_chunk(chunk_idx)
        chunks.append(RetrievedChunk(
            chunk_id=corpus_chunk.chunk_id,
            doc_id=corpus_chunk.doc_id,
            section=corpus_chunk.section,
            text=corpus_chunk.text,
            dense_score=dense_map.get(chunk_idx, 0.0),
            sparse_score=sparse_map.get(chunk_idx, 0.0),
            fused_score=fused_score,
            sub_query_id=sub_query.id,
        ))

    latency_ms = (time.time() - start_time) * 1000

    # Emit retrieval_completed
    telemetry_bus.emit(RetrievalEvent(
        event="retrieval_completed",
        session_id=session_id,
        turn_id=turn_id,
        sub_query_id=sub_query.id,
        mode="hybrid",
        latency_ms=latency_ms,
        candidate_count=len(chunks),
    ))

    return chunks


def retrieve_and_fuse(
    sub_queries: list[SubQuery],
    session_id: str,
    turn_id: str,
    top_k: int | None = None,
    index: CorpusIndex | None = None,
) -> dict[str, list[RetrievedChunk]]:
    """
    Run hybrid retrieval for ALL sub-queries in PARALLEL using a thread pool,
    then deduplicate within each sub-query's result set.

    Fusion is done WITHIN each sub-intent's evidence set independently —
    this prevents a strong match for one sub-intent from crowding out a
    weaker but correct match for another sub-intent.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[str, list[RetrievedChunk]] = {}

    def _retrieve_one(sq: SubQuery) -> tuple[str, list[RetrievedChunk]]:
        chunks = hybrid_retrieve(sq, session_id, turn_id, top_k, index)
        # Filter by minimum relevance
        chunks = [c for c in chunks if c.fused_score >= RETRIEVAL_MIN_RELEVANCE / 100]
        # Dedup within this sub-query's results
        chunks = deduplicate_chunks(chunks)
        # Emit fusion event
        telemetry_bus.emit(FusionEvent(
            session_id=session_id,
            turn_id=turn_id,
            sub_query_id=sq.id,
            dedup_count=0,
            final_ranked_chunk_ids=[c.chunk_id for c in chunks[:10]],
        ))
        return sq.id, chunks

    # Run all sub-queries in parallel (FAISS/BM25 are read-only, thread-safe)
    with ThreadPoolExecutor(max_workers=min(len(sub_queries), 4)) as executor:
        futures = {executor.submit(_retrieve_one, sq): sq for sq in sub_queries}
        for future in as_completed(futures):
            sq_id, chunks = future.result()
            results[sq_id] = chunks

    return results
