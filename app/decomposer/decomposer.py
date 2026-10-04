"""
Streaming Live RAG — Multi-Intent Decomposer
Decomposes compound utterances into parallel, independently-retrievable sub-queries.
One structured LLM call per RETRIEVE trigger (not per chunk).
Includes validation pass: merges near-duplicate sub-queries, caps at MAX_SUB_QUERIES.
"""
from __future__ import annotations
import json
import re
import time
import numpy as np
from typing import Optional

from app.config import MAX_SUB_QUERIES, GROQ_API_KEY, LLM_MODEL
from app.models import SubQuery, DecompositionEvent, TriggerType
from app.retrieval.corpus import corpus_index
from app.telemetry.bus import telemetry_bus


DECOMPOSITION_PROMPT = """You are a query decomposer for a retrieval-augmented generation system.
Given a user utterance, break it into independent, search-ready sub-queries.

Rules:
1. Each sub-query should be a self-contained search query that can retrieve relevant documents independently.
2. Remove conversational filler (um, so basically, actually, etc.).
3. Resolve pronouns using context if available.
4. Each sub-query should target ONE distinct information need.
5. If the utterance has only ONE intent, return a single sub-query.
6. Do NOT over-fragment: keep closely related aspects (e.g., "venue and its capacity") as one query.
7. Maximum {max_queries} sub-queries.

Return ONLY a JSON array of objects with "text" and "entities" fields.

Example:
Input: "I need a venue for 30 people in Pune, what's the cancellation policy, and is catering available?"
Output: [
  {{"text": "venue capacity 30 attendees Pune", "entities": ["venue", "30", "Pune"]}},
  {{"text": "cancellation policy venue booking", "entities": ["cancellation", "policy"]}},
  {{"text": "catering availability Pune venue", "entities": ["catering", "Pune"]}}
]

Input: "What's the password rotation policy for admin accounts?"
Output: [
  {{"text": "password rotation policy admin accounts", "entities": ["password", "rotation", "admin"]}}
]

Now decompose this utterance:
Input: "{utterance}"
Output:"""


def _call_llm_decompose(utterance: str) -> list[dict]:
    """Call Groq LLM for structured decomposition. Returns parsed sub-queries."""
    try:
        from groq import Groq

        if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
            # Fallback to rule-based decomposition
            return _rule_based_decompose(utterance)

        client = Groq(api_key=GROQ_API_KEY)
        prompt = DECOMPOSITION_PROMPT.format(
            max_queries=MAX_SUB_QUERIES,
            utterance=utterance
        )

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )

        content = response.choices[0].message.content.strip()
        print(f"[Decomposer] LLM Raw Output: {content}")
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r"\[.*\]", content, re.DOTALL)
        json_str = json_match.group() if json_match else content
        # Remove trailing commas
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
        try:
            raw_queries = json.loads(json_str)
            # If the LLM was lazy and just returned a single query for a multi-intent utterance, fall back!
            if len(raw_queries) <= 1 and _looks_multi_intent(utterance):
                print("[Decomposer] LLM returned 1 query but it looks multi-intent. Falling back to rule-based.")
                return _rule_based_decompose(utterance)
            return raw_queries
        except json.JSONDecodeError:
            # Fallback to regex extraction
            raw_queries = []
            for match in re.finditer(r'\{[^{}]*\}', json_str):
                obj_str = match.group()
                text_match = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', obj_str)
                entities_match = re.search(r'"entities"\s*:\s*\[(.*?)\]', obj_str)
                if text_match:
                    text = text_match.group(1).replace('\\"', '"').replace('\\n', '\n')
                    entities = []
                    if entities_match:
                        ent_str = entities_match.group(1)
                        entities = [e.strip('"\' ') for e in ent_str.split(',') if e.strip('"\' ')]
                    raw_queries.append({"text": text, "entities": entities})
            if len(raw_queries) <= 1 and _looks_multi_intent(utterance):
                return _rule_based_decompose(utterance)
            return raw_queries

    except Exception as e:
        print(f"[Decomposer] LLM call failed: {e}, falling back to rule-based")
        return _rule_based_decompose(utterance)


def _rule_based_decompose(utterance: str) -> list[dict]:
    """
    Fallback rule-based decomposition when LLM is unavailable.
    Splits on coordinating conjunctions between question-like clauses.
    """
    # Clean up filler
    text = re.sub(r"\b(um|uh|so basically|you know|like)\b", "", utterance, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()

    # Try splitting on ", and " / ", what" / "? " / "... " patterns
    parts = re.split(
        r"(?:,\s*and\s+(?=what|is|are|can|does|do|how|where|when))|"
        r"(?:,\s*(?=what|is|are|can|does|do|how|where|when))|"
        r"(?:\?\s*(?:and\s+)?(?=what|is|are|can|does|do|how|where|when))|"
        r"(?:\.{2,}\s*(?:and\s+)?(?=what|is|are|can|does|do|how|where|when))",
        text, flags=re.I
    )

    # Filter out empty/trivial parts
    parts = [p.strip().rstrip("?.,").strip() for p in parts if p.strip() and len(p.strip()) > 5]

    if not parts:
        parts = [text]

    result = []
    for part in parts[:MAX_SUB_QUERIES]:
        entities = _extract_query_entities(part)
        result.append({"text": part, "entities": entities})

    return result


def _extract_query_entities(text: str) -> list[str]:
    """Extract key entities from a query fragment."""
    entities = []
    # Numbers with units
    for m in re.finditer(r"\b(\d+)\s*(people|attendees|persons|seats|days|hours)\b", text, re.I):
        entities.append(m.group(0))
    # Location names
    for m in re.finditer(r"\b(Pune|Bangalore|Bengaluru|Mumbai|Delhi|Hyderabad|Chennai)\b", text, re.I):
        entities.append(m.group(0))
    # Key domain nouns
    for m in re.finditer(r"\b(venue|catering|cancel\w+|policy|expense|travel|password|leave|resign\w+|gym|fitness|parking)\b", text, re.I):
        entities.append(m.group(0))
    return list(set(entities))


def _merge_similar_subqueries(
    sub_queries: list[SubQuery],
    threshold: float = 0.9,
) -> tuple[list[SubQuery], list[str]]:
    """
    Merge sub-queries with high pairwise cosine similarity (>threshold).
    Returns (merged_list, merge_event_descriptions).
    """
    if len(sub_queries) <= 1:
        return sub_queries, []

    merge_events = []

    # Compute embeddings for all sub-queries
    try:
        texts = [sq.text for sq in sub_queries]
        embeddings = [corpus_index.embed_query(t) for t in texts]

        merged_indices = set()
        result = []

        for i in range(len(sub_queries)):
            if i in merged_indices:
                continue

            current = sub_queries[i]
            current_emb = embeddings[i]

            for j in range(i + 1, len(sub_queries)):
                if j in merged_indices:
                    continue

                sim = float(np.dot(current_emb, embeddings[j]))
                if sim > threshold:
                    # Merge j into i
                    merged_indices.add(j)
                    # Combine text (keep the longer one)
                    if len(sub_queries[j].text) > len(current.text):
                        current = SubQuery(
                            id=current.id,
                            text=sub_queries[j].text,
                            entities=list(set(current.entities + sub_queries[j].entities)),
                        )
                    else:
                        current = SubQuery(
                            id=current.id,
                            text=current.text,
                            entities=list(set(current.entities + sub_queries[j].entities)),
                        )
                    merge_events.append(
                        f"Merged sq '{sub_queries[j].text}' into '{current.text}' (sim={sim:.3f})"
                    )

            result.append(current)

        return result, merge_events

    except Exception:
        # If embedding fails, skip merging
        return sub_queries, []


def decompose_utterance(
    utterance: str,
    session_id: str,
    turn_id: str,
    trigger_type: TriggerType = TriggerType.PROVISIONAL,
) -> list[SubQuery]:
    """
    Main entry point: decompose an utterance into sub-queries.
    Uses LLM for multi-intent, rule-based for single-intent.
    """
    start_time = time.time()

    # For single/provisional triggers with no multi-intent signals, skip LLM
    if trigger_type == TriggerType.PROVISIONAL and not _looks_multi_intent(utterance):
        entities = _extract_query_entities(utterance)
        sub_queries = [SubQuery(text=utterance.strip(), entities=entities)]
    elif trigger_type == TriggerType.LATE_DETAIL:
        # Late detail: single sub-query targeting the delta
        entities = _extract_query_entities(utterance)
        sub_queries = [SubQuery(text=utterance.strip(), entities=entities)]
    else:
        # Multi-intent or ambiguous: use LLM/rule-based decomposition
        raw_queries = _call_llm_decompose(utterance)
        sub_queries = [
            SubQuery(
                text=q.get("text", utterance),
                entities=q.get("entities", []),
            )
            for q in raw_queries
        ]

    # Validation: merge near-duplicates
    sub_queries, merge_events = _merge_similar_subqueries(sub_queries)

    # Cap at MAX_SUB_QUERIES
    sub_queries = sub_queries[:MAX_SUB_QUERIES]

    # Ensure at least one sub-query
    if not sub_queries:
        sub_queries = [SubQuery(text=utterance.strip(), entities=_extract_query_entities(utterance))]

    # Emit telemetry
    telemetry_bus.emit(DecompositionEvent(
        session_id=session_id,
        turn_id=turn_id,
        raw_utterance=utterance,
        sub_queries=[{"id": sq.id, "text": sq.text, "entities": sq.entities} for sq in sub_queries],
        merge_events=merge_events,
    ))

    return sub_queries


def _looks_multi_intent(text: str) -> bool:
    """Quick heuristic check for multi-intent signals."""
    indicators = [
        r"\b(and\s+(?:also|what|is|are|can|does|do|how|where|when))\b",
        r",\s*(what|is|are|can|does|do|how|where|when)\b",
        r"\?\s*(and|also|what|plus)\b",
    ]
    for pattern in indicators:
        if re.search(pattern, text, re.I):
            return True
    return text.count("?") >= 2
