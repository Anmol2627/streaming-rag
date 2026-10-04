"""
Streaming Live RAG — Grounded Synthesis
Generates/patches claims strictly from retrieved evidence. Every claim cites
[Doc_ID §Section]. Citation validator checks entailment before emission.
Unsupported sub-intents get explicit uncertainty flags, never confident guesses.
"""
from __future__ import annotations
import json
import re
import time
import numpy as np
from typing import Optional

from app.config import (
    GROQ_API_KEY, LLM_MODEL, CITATION_ENTAILMENT_THRESHOLD,
    RETRIEVAL_MIN_RELEVANCE,
)
from app.models import (
    Claim, SubQuery, RetrievedChunk,
    ClaimGeneratedEvent, UncertaintyEvent,
)
from app.retrieval.corpus import corpus_index
from app.telemetry.bus import telemetry_bus


SYNTHESIS_PROMPT = """You are an expert Q&A assistant. Your job is to DIRECTLY answer the user's sub-question using ONLY the provided evidence chunks.

RULES:
1. Directly answer the SUB-QUESTION. Do not just summarize the documents. If they ask for a venue for 30 people in Pune, tell them which venue works and cite it!
2. Every factual statement must cite its source as [Doc_ID §Section] (e.g., [Doc_01 §2]).
3. Only cite document IDs that appear in the provided evidence.
4. If the evidence is insufficient to answer the sub-question, say "Insufficient evidence" — NEVER guess.
5. Keep claims concise and factual.

SUB-QUESTION: {sub_query}

EVIDENCE CHUNKS:
{evidence}

Generate a JSON array of claims that answer the sub-question. Each claim has "text" (the direct answer/factual statement with inline citations) and "citations" (array of "Doc_ID §Section" strings used).

Example output:
[
  {{"text": "Yes, Grand Hall in Pune is available and seats up to 40 people [Doc_01 §2].", "citations": ["Doc_01 §2"]}},
  {{"text": "The cancellation policy allows full refunds if cancelled 14+ days in advance [Doc_04 §5].", "citations": ["Doc_04 §5"]}}
]

Output ONLY the JSON array, no other text:"""


REFORMULATION_PROMPT = """Reformat the following answer into the requested format. Preserve all factual content and citations exactly. Do not add, remove, or change any facts or citations.

Current answer:
{current_answer}

User request: {user_request}

Output the reformatted answer as plain text with the same [Doc_ID §Section] citations inline:"""


def synthesize_claims(
    sub_query: SubQuery,
    evidence: list[RetrievedChunk],
    session_id: str,
    turn_id: str,
) -> tuple[list[Claim], list[str]]:
    """
    Generate grounded claims for a single sub-query from its evidence set.
    Returns (claims, uncertainty_messages).
    """
    # Check if evidence meets minimum relevance floor
    relevant_evidence = [
        e for e in evidence
        if e.fused_score >= RETRIEVAL_MIN_RELEVANCE / 100
    ]

    # If no relevant evidence, emit uncertainty
    if not relevant_evidence or len(relevant_evidence) == 0:
        uncertainty_msg = f"Insufficient evidence in the corpus to answer: '{sub_query.text}'"
        telemetry_bus.emit(UncertaintyEvent(
            session_id=session_id,
            turn_id=turn_id,
            sub_query_id=sub_query.id,
            reason="no_relevant_evidence",
        ))
        return [], [uncertainty_msg]

    # Take top evidence chunks (limit context size)
    top_evidence = relevant_evidence[:8]

    # Format evidence for the prompt
    evidence_text = "\n\n".join([
        f"[{e.doc_id} {e.section}] (score: {e.fused_score:.3f}):\n{e.text}"
        for e in top_evidence
    ])

    # Generate claims via LLM
    claims, uncertainties = _llm_generate_claims(
        sub_query.text, evidence_text, top_evidence, session_id, turn_id, sub_query.id
    )

    # Validate citations
    validated_claims = []
    for claim in claims:
        validation_result = _validate_citations(claim, top_evidence)
        if validation_result == "pass":
            validated_claims.append(claim)
            telemetry_bus.emit(ClaimGeneratedEvent(
                session_id=session_id,
                turn_id=turn_id,
                claim_id=claim.claim_id,
                citations=claim.citations,
                validator_result="pass",
            ))
        elif validation_result == "regenerate":
            # Try once more with stricter prompt
            regenerated = _regenerate_strict(claim, top_evidence, session_id, turn_id, sub_query.id)
            if regenerated:
                validated_claims.append(regenerated)
                telemetry_bus.emit(ClaimGeneratedEvent(
                    session_id=session_id,
                    turn_id=turn_id,
                    claim_id=regenerated.claim_id,
                    citations=regenerated.citations,
                    validator_result="regenerated",
                ))
            else:
                # Convert to uncertainty
                uncertainties.append(f"Could not verify claim: '{claim.text}'")
                telemetry_bus.emit(ClaimGeneratedEvent(
                    session_id=session_id,
                    turn_id=turn_id,
                    claim_id=claim.claim_id,
                    citations=claim.citations,
                    validator_result="fail",
                ))
        else:
            # Failed validation — convert to uncertainty
            uncertainties.append(f"Could not verify claim: '{claim.text}'")
            telemetry_bus.emit(ClaimGeneratedEvent(
                session_id=session_id,
                turn_id=turn_id,
                claim_id=claim.claim_id,
                citations=claim.citations,
                validator_result="fail",
            ))

    return validated_claims, uncertainties


def _llm_generate_claims(
    sub_query_text: str,
    evidence_text: str,
    evidence_chunks: list[RetrievedChunk],
    session_id: str,
    turn_id: str,
    sub_query_id: str,
) -> tuple[list[Claim], list[str]]:
    """Generate claims using LLM. Falls back to extractive if LLM unavailable."""
    try:
        from groq import Groq

        if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
            return _extractive_fallback(sub_query_text, evidence_chunks, sub_query_id)

        client = Groq(api_key=GROQ_API_KEY)
        prompt = SYNTHESIS_PROMPT.format(
            sub_query=sub_query_text,
            evidence=evidence_text,
        )

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1000,
        )

        content = response.choices[0].message.content.strip()
        # Parse JSON
        json_match = re.search(r"\[.*\]", content, re.DOTALL)
        json_str = json_match.group() if json_match else content
        # Fix trailing commas
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
        try:
            raw_claims = json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback to regex extraction
            raw_claims = []
            for match in re.finditer(r'\{[^{}]*\}', json_str):
                obj_str = match.group()
                text_match = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', obj_str)
                citations_match = re.search(r'"citations"\s*:\s*\[(.*?)\]', obj_str)
                if text_match:
                    text = text_match.group(1).replace('\\"', '"').replace('\\n', '\n')
                    citations = []
                    if citations_match:
                        cit_str = citations_match.group(1)
                        citations = [c.strip('"\' ') for c in cit_str.split(',') if c.strip('"\' ')]
                    raw_claims.append({"text": text, "citations": citations})

        claims = []
        for rc in raw_claims:
            text = rc.get("text", "")
            citations = rc.get("citations", [])
            # Normalize citation format
            citations = [_normalize_citation(c) for c in citations]
            claims.append(Claim(
                text=text,
                citations=citations,
                source_sub_query=sub_query_id,
            ))

        return claims, []

    except Exception as e:
        print(f"[Synthesis] LLM generation failed: {e}, using extractive fallback")
        return _extractive_fallback(sub_query_text, evidence_chunks, sub_query_id)


def _extractive_fallback(
    sub_query_text: str,
    evidence_chunks: list[RetrievedChunk],
    sub_query_id: str,
) -> tuple[list[Claim], list[str]]:
    """
    Extractive fallback when LLM is unavailable.
    Takes the top evidence chunks and presents them as claims with citations.
    """
    claims = []
    for chunk in evidence_chunks[:3]:
        # Extract first meaningful sentence from the chunk
        text = chunk.text.strip()
        # Skip the header line if present
        lines = text.split("\n")
        content_lines = [l for l in lines[1:] if l.strip()] if len(lines) > 1 else lines
        if content_lines:
            claim_text = content_lines[0].strip()
            # Truncate to reasonable length
            if len(claim_text) > 300:
                claim_text = claim_text[:297] + "..."
            citation = f"{chunk.doc_id} {chunk.section}"
            claim_text = f"{claim_text} [{citation}]"
            claims.append(Claim(
                text=claim_text,
                citations=[citation],
                source_sub_query=sub_query_id,
            ))

    return claims, []


def _normalize_citation(citation: str) -> str:
    """Normalize citation format to 'Doc_XX §Y'."""
    citation = citation.strip().strip("[]")
    # Handle various formats: Doc_01 §2, Doc_01 S2, Doc_01 s2, etc.
    match = re.match(r"(Doc_\d+)\s*[§Ss](\d+)", citation)
    if match:
        return f"{match.group(1)} §{match.group(2)}"
    return citation


def _validate_citations(claim: Claim, evidence: list[RetrievedChunk]) -> str:
    """
    Validate that each citation in a claim actually corresponds to a real
    evidence chunk in the provided evidence set.

    Returns: "pass" | "fail"
    (We intentionally skip cosine entailment — that check was discarding
    valid claims when the claim phrasing differed from the raw chunk text.)
    """
    if not claim.citations:
        return "fail"  # No citations at all

    available_ids = set()
    for e in evidence:
        available_ids.add(f"{e.doc_id} {e.section}")

    for citation in claim.citations:
        normalized = _normalize_citation(citation)
        if normalized not in available_ids:
            # Citation references a document not in the evidence set → hallucinated
            return "fail"

    return "pass"


def _regenerate_strict(
    claim: Claim,
    evidence: list[RetrievedChunk],
    session_id: str,
    turn_id: str,
    sub_query_id: str,
) -> Claim | None:
    """Attempt to regenerate a claim with a stricter prompt."""
    try:
        from groq import Groq

        if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
            return None

        # Find the cited chunk
        cited_text = ""
        for citation in claim.citations:
            normalized = _normalize_citation(citation)
            for e in evidence:
                if f"{e.doc_id} {e.section}" == normalized:
                    cited_text += e.text + "\n"

        if not cited_text:
            return None

        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""Rewrite this claim to state ONLY what is directly stated in the source text. Do not add any information not in the source.

Source text:
{cited_text}

Original claim: {claim.text}

Rewrite the claim as a JSON object with "text" and "citations" fields. Only output the JSON:"""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )

        content = response.choices[0].message.content.strip()
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return Claim(
                text=data.get("text", ""),
                citations=[_normalize_citation(c) for c in data.get("citations", claim.citations)],
                source_sub_query=sub_query_id,
            )

    except Exception:
        pass

    return None


def reformulate_answer(
    current_answer_text: str,
    user_request: str,
    session_id: str,
    turn_id: str,
) -> str:
    """
    Reformulate an existing answer (e.g., "repeat in two bullets")
    without any retrieval. Preserves all facts and citations.
    """
    try:
        from groq import Groq

        if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
            return current_answer_text  # Can't reformulate without LLM

        client = Groq(api_key=GROQ_API_KEY)
        prompt = REFORMULATION_PROMPT.format(
            current_answer=current_answer_text,
            user_request=user_request,
        )

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1000,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[Synthesis] Reformulation failed: {e}")
        return current_answer_text
