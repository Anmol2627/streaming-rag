# Streaming Live RAG — Development / Demo Corpus Package

This is **our own** development, benchmarking, and demo corpus — not the Samsung grading
corpus. Read this file's warning below before touching the application code.

## A. Corpus Design Philosophy

This corpus is a self-authored **Corporate Operations Knowledge Base** (~24 documents, 153
citable sections), built specifically to exercise every architectural behavior the Streaming
Live RAG system must demonstrate: early retrieval, multi-intent decomposition, late-detail
refinement, no-retrieval suppression, grounded uncertainty, deduplication, conflicting evidence,
and hybrid dense/sparse retrieval. It is intentionally sized to be realistic and debuggable —
large enough for genuine retrieval competition between candidate chunks, small enough that every
fact and every edge case is fully traceable by hand. Documents are grouped into six domains
(venues, catering, travel & expense, HR/employment, IT security, office facilities) chosen
because they naturally produce compound questions, evolving details, and policy documents that
legitimately update or conflict with one another — exactly the corpus properties the evaluation
gates are designed to test.

**This corpus authors deliberate edge cases** (near-duplicate passages, a genuine
document-versioning conflict, topics with zero coverage) rather than only "easy" content, because
a corpus with no hard cases cannot demonstrate the controller, deduplication, grounding, or
conflict-handling logic at all.

## ⚠️ Critical: Two Separate Corpora — Do Not Conflate

1. **Samsung Grading Corpus** — hidden, private, held out, entirely outside our control. The
   application's ingestion, chunking, retrieval, metadata schema, and controller logic must be
   fully **corpus-agnostic**: it must accept an arbitrary folder of documents and build all
   required indexes automatically, with zero assumptions baked in about document names, IDs, or
   content from the corpus described in this package.
2. **This corpus** (`/corpus/documents/`) exists **only** for our own development, benchmarking,
   screenshots, and demo video. It is processed through the exact same generic ingestion
   pipeline the application uses for any corpus — it receives no special-cased handling anywhere
   in the application code. None of these documents, benchmark cases, or expected answers may be
   hardcoded into the application; the benchmark harness (Section on Benchmarks below) is the
   only place this corpus's content should ever be referenced by ID.

## B. Document Inventory

See `/corpus/metadata/document_inventory.md` for the full table (Doc_ID, filename, purpose,
section count, key facts, relationships to other documents).

## C. Benchmark Taxonomy

`/corpus/benchmarks/benchmark_suite.json` contains 48 scenarios across 8 categories:

| Category | Count | Tests |
|---|---|---|
| `early_retrieval` | 7 | Controller fires RETRIEVE before utterance end (G2) |
| `no_retrieval` | 7 | Presentation-only follow-ups correctly suppress retrieval |
| `multi_intent` | 7 | Compound utterances decomposed into correct sub-intents (G3) |
| `late_detail` | 7 | Targeted claim updates without full-corpus re-search (G5) |
| `uncertainty` | 6 | Explicit uncertainty on thin/absent corpus coverage (G4) |
| `duplicate_retrieval` | 4 | Near-duplicate passages correctly deduplicated (and not over-merged) |
| `conflicting_evidence` | 4 | Genuine document conflicts surfaced rather than silently resolved |
| `hybrid_retrieval` | 6 | Dense-only / sparse-only / both-required retrieval cases |

Each scenario record includes: `scenario_id`, `category`, `transcript_chunks`,
`expected_controller_decisions`, `trigger_type_if_retrieve`, `expected_sub_intents`,
`expected_retrieved_doc_ids`, `expected_relevant_sections`, `expected_answer_summary`,
`expected_citations`, `answer_versioning_expected`, `affected_claims`,
`expected_telemetry_events`, `expected_grounding_behavior`.

## D. Golden Demo Scenario List

See `/corpus/golden_scenarios/golden_demo_scenarios.md` for full detail. Five scenarios, ready
for the hackathon demo video:

1. Early Retrieval (Bangalore venue, 100 people)
2. Multi-Intent Parallel Retrieval (Pune venue + cancellation + catering)
3. Late Detail → Targeted Refinement (30 → 50 people, Answer v1 → v2)
4. No-Retrieval Conversational Follow-Up ("repeat that in two bullets")
5. Insufficient Evidence → Explicit Uncertainty (Bangalore gym question)

## E. Folder Structure

```
/corpus
  /documents          24 markdown documents, Doc_01–Doc_24, each with explicit
                       # Doc_ID / ## §N Section Name headers mapping 1:1 to the
                       [Doc_ID §Section] citation format.
  /metadata
    document_inventory.md                 Deliverable B — full document inventory table
    duplicates_conflicts_thin_coverage.md Deliverable D-part — near-duplicates, genuine
                                           conflicts, thin-coverage/uncertainty topics,
                                           and the hybrid-retrieval design notes
  /benchmarks
    benchmark_suite.json                  48 scenarios, machine-readable, for the
                                           benchmark harness described in the master
                                           build specification
  /golden_scenarios
    golden_demo_scenarios.md              5 fully worked demo scenarios with exact
                                           timing, decisions, and citations
  README.md                               This file
```

## Using This Package

The coding agent should point the application's generic ingestion pipeline at
`/corpus/documents/` to build the development-time indexes, then run
`/corpus/benchmarks/benchmark_suite.json` through the benchmark harness to compute
development-time approximations of gates G2–G6 (G1 and G6 also depend on the packaging/telemetry
work described in the master build specification). The golden scenarios in
`/corpus/golden_scenarios/` are the ones to rehearse for the final demo video — nothing here
should be copied into application source code.
