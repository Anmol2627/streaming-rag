# Duplicates, Conflicts, Thin-Coverage & Hybrid-Retrieval Design Notes

## Near-Duplicate Passages (Deduplication test material)
1. **Doc_12 §3** (Employee Handbook — Onboarding: "IT Security Training Requirement") and
   **Doc_17 §3** (IT Security — Device Onboarding: "IT Security Training Requirement") both state
   new employees/devices must complete security training within 5 working days, phrased
   independently. A query about security-training deadlines should retrieve both, and the
   fusion/dedup stage should present this as one fact citing both sources rather than two
   redundant claims.
2. **Doc_19 §6** and **Doc_20 §6** (Visitor Access Procedure, Bangalore and Pune respectively)
   are near-identical in content ("2 hours in advance", "government-issued photo ID") but are
   legitimately two separate campus policies, not a single duplicated fact — a good negative
   control for over-merging (they should be cited separately, per campus, if the query is
   campus-specific; merged only if the query is genuinely campus-agnostic).

## Genuinely Conflicting Evidence
1. **Doc_14 §2** (Employee Handbook v3: 30-day resignation notice period, all employees) vs.
   **Doc_15 §2** (HR Policy Update Memo 2026: 45-day notice period, all employees, explicitly
   "superseding Section 2 of the Employee Handbook"). This is a realistic, undisclosed-in-place
   document-drift conflict: the handbook has not yet been edited to reflect the memo. The system
   should surface both, note the memo's later effective date and explicit superseding language,
   and prefer the memo's answer while disclosing the discrepancy — not silently pick one without
   acknowledgment.
2. **Doc_08 §3** (Domestic Travel Policy: 30-day receipt submission window) vs. **Doc_11 §2**
   (2026 Addendum Memo: window reduced to 15 days). Resolvable by the effective-date transition
   clause in **Doc_11 §4** (applies only to trips completed on/after the next fiscal quarter),
   making this a harder, date-conditional conflict rather than a flat contradiction.

## Thin-Coverage Topics (Uncertainty test material — intentionally absent from the corpus)
- Sabbatical / extended unpaid leave beyond the categories in Doc_13 — not covered anywhere.
- Wellness/mental-health stipend (distinct from the Doc_18 §4 home-office equipment stipend) — not covered.
- Gym/fitness facility at the **Bangalore** campus specifically — Doc_19 lists parking, cafeteria,
  meeting rooms, mailroom, visitor access, and lost & found, but no fitness facility, in deliberate
  contrast to Doc_20 §5 (Pune campus, which does have one). Tests that the system does not
  incorrectly borrow Doc_20's fitness-room fact when asked about Bangalore.
- Relocation assistance for international/permanent transfers — Doc_21 covers business-travel visa
  support only, which is adjacent but must not be conflated with relocation assistance.
- EV charging stations at either campus — not covered.
- Pet-friendly office policy — not covered.

## Hybrid Retrieval Design (Dense vs. Sparse vs. Both)
- **Sparse/BM25-friendly** (exact codes/numbers, low semantic ambiguity): Doc_16 §3 ("SEC-014",
  "90 days", "60 days"); Doc_03 §2 ("Room B-204", "180 attendees").
- **Dense/semantic-friendly** (abstract phrasing, no exact keyword overlap with likely queries):
  Doc_18 §1 (remote-work philosophy, no literal "remote work" keyword match to a paraphrased
  query); Doc_03 §5 (ambience/atrium description, no literal keyword match to a query about
  "atmosphere for informal networking").
- **Both required**: queries combining an exact code/number lookup with a semantic paraphrase in
  the same turn (see benchmark scenarios HYB-03 and HYB-06) — the clearest demonstration of why
  dense+sparse+RRF outperforms either alone.
