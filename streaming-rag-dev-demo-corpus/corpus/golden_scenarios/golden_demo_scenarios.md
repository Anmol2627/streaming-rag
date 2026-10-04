# Golden Demo Scenarios — Streaming Live RAG

Five complete, hackathon-demo-ready scenarios. Each includes exact transcript timing, controller
decisions, retrieved evidence, and exact expected citations. Use these directly for the system
demonstration video and live judged walkthrough.

---

## Golden Demo 1 — Early Retrieval

**Transcript stream:**

| Timestamp | Chunk | Controller Decision |
|---|---|---|
| 0.0s | "I need to book a venue in Bangalore..." | WAIT — intent incomplete, no capacity/date yet |
| 0.9s | "...for a hundred people..." | **PROVISIONAL RETRIEVE** — capacity entity now present, location already given. Sub-query: "Bangalore venue capacity 100" |
| 1.8s | "...sometime next month for a hackathon." | Utterance End → Synthesize |

**What the judge should notice:** retrieval visibly starts at 0.9s, ~0.9 seconds before the
utterance actually ends at 1.8s — a clean, on-screen timestamp gap.

**Retrieved evidence:** Doc_03 §2 (Room B-204, 100 seated theatre-style), Doc_03 §7 (15 business
day booking lead time).

**Expected citations:** `[Doc_03 §2]`, `[Doc_03 §7]`

**Expected answer:** "Room B-204 at the Bangalore Tech Park Convention Wing seats 100
theatre-style, which fits your group exactly. Bookings for this room require 15 business days
advance notice, so plan the request accordingly for next month."

---

## Golden Demo 2 — Multi-Intent Parallel Retrieval

**User utterance:** *"I need a venue for 30 people in Pune, what's the cancellation policy, and
is catering available?"*

**Transcript timeline:**

| Timestamp | Chunk | Controller Decision |
|---|---|---|
| 0.0s | "I need a venue..." | WAIT |
| 0.8s | "...for 30 people in Pune..." | PROVISIONAL RETRIEVE — sub-query: "venue capacity 30 Pune" |
| 1.6s | "...what's the cancellation policy, and is catering available?" | **DECOMPOSE & PARALLEL RETRIEVE** into 3 sub-queries |
| 2.1s | [Utterance End] | Synthesize |

**Sub-queries dispatched in parallel:**
1. `sq1`: venue capacity 30 Pune
2. `sq2`: cancellation policy venue booking
3. `sq3`: catering availability Pune venue

**Retrieved evidence:** Doc_01 §2 (Grand Hall, fits 30), Doc_04 §5 (uniform 14-day cancellation
rule), Doc_01 §5 / Doc_05 §7 (Spice Route catering coordination).

**Expected citations:** `[Doc_01 §2]`, `[Doc_04 §5]`, `[Doc_01 §5]`, `[Doc_05 §7]`

**Expected answer:** "Grand Hall Conference Center in Pune comfortably fits 30 attendees
(theatre-style capacity 40). Cancellations made more than 14 days before the event get a full
deposit refund; within 14 days, 50% of the deposit is forfeited. Spice Route catering can be
brought in with 48 hours' notice to the venue coordinator."

**What the judge should notice:** three sub-queries fan out in parallel on the dashboard, and the
final answer unifies all three without dropping or conflating any sub-intent.

---

## Golden Demo 3 — Late Detail → Targeted Refinement

**Step 1 — Initial request:** *"I need a venue for 30 people in Pune, what's the cancellation
policy, and is catering available?"*
→ **Answer v1** (identical to Golden Demo 2): venue = Grand Hall, cancellation = 14-day rule,
catering = Spice Route available.
Citations: `[Doc_01 §2]`, `[Doc_04 §5]`, `[Doc_05 §7]`

**Step 2 — Late detail arrives:** *"Actually, make it 50 people."*

**Affected-claim detection:** 50 exceeds Grand Hall's maximum standing capacity of 45 (Doc_01
§2) → the venue/capacity claim is flagged as affected. The cancellation claim (sourced from the
venue-agnostic Doc_04 §5) and the catering claim (Spice Route also serves Pune venues per Doc_05
§7) are **not** flagged.

**Targeted retrieval:** searches only for a Pune venue with capacity ≥ 50 → retrieves Doc_02 §2
(Riverside, 80 seated / 100 standing).

**Step 3 — Answer v2:**
- **Changed claim:** venue recommendation now Riverside Conference Center (was Grand Hall).
- **Unchanged claims (byte-identical to v1):** cancellation policy claim, catering claim.

**Expected citations, v2:** `[Doc_02 §2]` (replaces `[Doc_01 §2]`), `[Doc_04 §5]` (unchanged),
`[Doc_05 §7]` (unchanged)

**Expected answer (v2):** "For 50 people, Riverside Conference Center in Pune is the better fit —
its main hall seats up to 80 theatre-style. The cancellation policy and Spice Route catering
availability from your original question are unchanged."

**What the judge should notice:** the version counter increments from 1 to 2, only the venue
claim visibly updates, and no full-corpus re-search event appears in the telemetry log for this
turn.

---

## Golden Demo 4 — No-Retrieval Conversational Follow-Up

Immediately following Golden Demo 2 or 3's delivered answer, the user says: *"Can you repeat
that in two bullet points?"*

**Controller decision:** NO-RETRIEVAL, `reason: presentation_restructure`.

**System behavior:** reformats the existing answer (whichever version is current) into two
bullet points, reusing the exact same citations. Zero new `retrieval_started` events fire.

**Expected output (reformatted from Golden Demo 2's answer):**
- Grand Hall Conference Center in Pune fits 30 attendees; cancellations more than 14 days out
  get a full deposit refund, within 14 days forfeit 50%.
- Spice Route catering can be arranged with 48 hours' notice.

**Citations (unchanged):** `[Doc_01 §2]`, `[Doc_04 §5]`, `[Doc_05 §7]`

**What the judge should notice:** the telemetry log shows a controller decision with zero
accompanying retrieval events — proof the system recognized this as presentation-only.

---

## Golden Demo 5 — Insufficient Evidence → Explicit Uncertainty

**User utterance:** *"Does the Bangalore office have a gym, and if so what are the hours?"*

**Controller decision:** RETRIEVE (provisional) — sub-query: "Bangalore office gym fitness
facility hours."

**Retrieval:** returns Doc_19 (Bangalore Office Facilities) in full — sections cover parking,
cafeteria, meeting rooms, mailroom, visitor access, and lost & found. No section mentions a gym
or fitness facility, and no chunk clears the minimum relevance floor for the "gym" concept.

**Grounding check:** the system must **not** substitute Doc_20 §5 (Pune campus fitness room) as
if it answered the Bangalore question — the two are different campuses and must not be
conflated.

**Expected output:** "The available information doesn't show a gym or fitness facility at the
Bangalore campus. (For reference, the Pune campus does have an on-site fitness room, but that's
a separate location.)"

**Expected citations:** none for the core (unanswerable) claim; `[Doc_20 §5]` only if the
optional Pune reference is included, clearly labeled as being about a different campus.

**What the judge should notice:** the system explicitly declines to guess, and the telemetry log
shows an `uncertainty_flagged` event rather than a fabricated citation — this is the moment that
most convincingly demonstrates G4 (zero hallucinated grounding) to a skeptical judge.
