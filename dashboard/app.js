/**
 * Streaming Live RAG — Dashboard Logic
 */

let currentSessionId = null;
let eventSource = null;
let chunkIndex = 0;
let streamingChunks = [];
let streamInterval = null;
let previousAnswerVersion = 0;

// Golden Demo Scenarios (Matches benchmark_suite.json precisely)
const SCENARIOS = {
    "ER-01": [
        "I need a venue in Bangalore...",
        "...for a hundred people...",
        "...sometime next month for a hackathon."
    ],
    "MI-01": [
        "I need a venue for 30 people in Pune...",
        "...what's the cancellation policy...",
        "...and is catering available?"
    ],
    "LD-01": [
        "I need a venue for 30 people in Pune, what's the cancellation policy, and is catering available?"
    ],
    "LD-02": [
        "Actually, make it 50 people."
    ],
    "NR-01": [
        "Can you repeat that in two bullet points?"
    ],
    "UN-01": [
        "Does the Bangalore office have a gym...",
        "...and if so what are the hours?"
    ]
};

// DOM Elements
const els = {
    btnNewSession: document.getElementById('btn-new-session'),
    btnLoadScenario: document.getElementById('btn-load-scenario'),
    btnSendChunk: document.getElementById('btn-send-chunk'),
    btnEndTurn: document.getElementById('btn-end-turn'),
    btnClearLog: document.getElementById('btn-clear-log'),
    scenarioSelect: document.getElementById('scenario-select'),
    chunkInput: document.getElementById('chunk-input'),
    connStatus: document.getElementById('conn-status'),
    sessionId: document.getElementById('session-id-display'),
    
    // Containers
    transcript: document.getElementById('transcript-container'),
    subQueries: document.getElementById('sub-queries-container'),
    evidence: document.getElementById('evidence-container'),
    answer: document.getElementById('answer-container'),
    telemetry: document.getElementById('telemetry-container'),
    
    // Controller Status
    ctrlDecision: document.getElementById('ctrl-decision'),
    ctrlReason: document.getElementById('ctrl-reason'),
    ctrlBuffer: document.getElementById('ctrl-buffer'),
    
    // Metrics
    sqCount: document.getElementById('sq-count'),
    ansVersion: document.getElementById('answer-version'),
    mTTFR: document.getElementById('m-ttfr'),
    mTTFT: document.getElementById('m-ttft'),
    mTokens: document.getElementById('m-tokens')
};

// ─── Initialization & Session Management ──────────────────────────────────

async function createNewSession() {
    try {
        const res = await fetch('/session', { method: 'POST' });
        const data = await res.json();
        currentSessionId = data.session_id;
        
        els.sessionId.textContent = currentSessionId.substring(0, 8);
        resetUI();
        connectSSE();
    } catch (e) {
        console.error("Failed to create session", e);
        els.connStatus.textContent = "Error";
    }
}

function connectSSE() {
    if (eventSource) eventSource.close();
    
    els.connStatus.textContent = "Connecting...";
    eventSource = new EventSource(`/session/${currentSessionId}/telemetry`);
    
    eventSource.onopen = () => {
        els.connStatus.textContent = "Connected";
        els.connStatus.classList.add('connected');
    };
    
    eventSource.onmessage = (e) => {
        const event = JSON.parse(e.data);
        if (event.event === "heartbeat") return;
        handleTelemetryEvent(event);
    };
    
    eventSource.onerror = () => {
        els.connStatus.textContent = "Disconnected";
        els.connStatus.classList.remove('connected');
    };
}

function resetUI() {
    els.transcript.innerHTML = '';
    els.subQueries.innerHTML = '<div class="placeholder-text">No queries dispatched yet.</div>';
    els.evidence.innerHTML = '<div class="placeholder-text">Waiting for retrieval...</div>';
    els.answer.innerHTML = '<div class="placeholder-text">Waiting for synthesis...</div>';
    els.telemetry.innerHTML = '';
    
    els.ctrlDecision.textContent = 'WAIT';
    els.ctrlDecision.className = 'state-value dec-WAIT';
    els.ctrlReason.textContent = '-';
    els.ctrlBuffer.textContent = '-';
    
    els.sqCount.textContent = '0';
    els.ansVersion.textContent = '0';
    els.mTTFR.textContent = '- ms';
    els.mTTFT.textContent = '- ms';
    els.mTokens.textContent = '-';
    
    chunkIndex = 0;
    previousAnswerVersion = 0;
}

// ─── Sending Data ─────────────────────────────────────────────────────────

async function sendChunk(text) {
    if (!currentSessionId) await createNewSession();
    
    const chunkTs = Date.now() / 1000;
    
    // Add to UI immediately
    addTranscriptBubble(text, chunkTs);
    
    try {
        const res = await fetch(`/session/${currentSessionId}/chunk`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                chunk_text: text,
                chunk_ts: chunkTs,
                chunk_index: chunkIndex++
            })
        });
        
        const decision = await res.json();
        updateControllerUI(decision);
        
    } catch (e) {
        console.error("Chunk send failed", e);
    }
}

async function endTurn() {
    if (!currentSessionId) return;
    
    // We send an empty chunks array to signal turn end. The backend uses the existing buffer.
    try {
        els.answer.innerHTML = '<div class="placeholder-text">Synthesizing...</div>';
        
        const res = await fetch(`/session/${currentSessionId}/turn`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chunks: [] })
        });
        
        const data = await res.json();
        renderAnswer(data);
        
    } catch (e) {
        console.error("Turn end failed", e);
        els.answer.innerHTML = '<div class="placeholder-text" style="color:red">Synthesis failed</div>';
    }
}

// ─── UI Updaters ──────────────────────────────────────────────────────────

function addTranscriptBubble(text, ts) {
    const div = document.createElement('div');
    div.className = 'chunk-bubble';
    const date = new Date(ts * 1000);
    const timeStr = `${date.getMinutes().toString().padStart(2, '0')}:${date.getSeconds().toString().padStart(2, '0')}.${date.getMilliseconds().toString().padStart(3, '0').substring(0, 1)}`;
    
    div.innerHTML = `<span class="chunk-ts">${timeStr}</span>${text}`;
    els.transcript.appendChild(div);
    els.transcript.scrollTop = els.transcript.scrollHeight;
}

function updateControllerUI(decision) {
    els.ctrlDecision.textContent = decision.decision;
    els.ctrlDecision.className = `state-value dec-${decision.decision}`;
    els.ctrlReason.textContent = decision.reason;
    els.ctrlBuffer.textContent = decision.buffer || "-";
}

function handleTelemetryEvent(event) {
    // 1. Add to log
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    const date = new Date(event.timestamp * 1000);
    const timeStr = `${date.getHours()}:${date.getMinutes().toString().padStart(2, '0')}:${date.getSeconds().toString().padStart(2, '0')}.${date.getMilliseconds().toString().padStart(3, '0').substring(0, 1)}`;
    
    let summaryData = "";
    if (event.event === "controller_decision") summaryData = `${event.decision} (${event.reason})`;
    else if (event.event === "retrieval_started") summaryData = `sq_id: ${event.sub_query_id}`;
    else if (event.event === "retrieval_completed") summaryData = `${event.candidate_count} chunks in ${event.latency_ms.toFixed(1)}ms`;
    else if (event.event === "decomposition") summaryData = `${event.sub_queries.length} queries generated`;
    else if (event.event === "answer_version") summaryData = `v${event.version} (Trigger: ${event.trigger})`;
    else if (event.event === "turn_summary") summaryData = `TTFR: ${event.time_to_first_retrieval_ms.toFixed(0)}ms, TTFT: ${event.time_to_first_token_ms.toFixed(0)}ms`;
    
    logEntry.innerHTML = `
        <span class="log-ts">[${timeStr}]</span>
        <span class="log-type">${event.event}</span>
        <span class="log-data">${summaryData}</span>
    `;
    els.telemetry.appendChild(logEntry);
    els.telemetry.scrollTop = els.telemetry.scrollHeight;

    // 2. Update Specific UI components based on event
    if (event.event === "decomposition") {
        renderSubQueries(event.sub_queries);
    } 
    else if (event.event === "turn_summary") {
        if (event.time_to_first_retrieval_ms > 0) {
            els.mTTFR.textContent = event.time_to_first_retrieval_ms.toFixed(0) + " ms";
        }
        els.mTTFT.textContent = event.time_to_first_token_ms.toFixed(0) + " ms";
        els.mTokens.textContent = event.total_tokens || 0;
    }
}

function renderSubQueries(queries) {
    if (!queries || queries.length === 0) return;
    
    els.sqCount.textContent = queries.length;
    els.subQueries.innerHTML = '';
    
    queries.forEach(sq => {
        const div = document.createElement('div');
        div.className = 'sq-card';
        div.innerHTML = `
            <div class="sq-id">${sq.id}</div>
            <div>${sq.text}</div>
            ${sq.entities && sq.entities.length ? `<div class="text-xs text-muted mt-2">Entities: ${sq.entities.join(', ')}</div>` : ''}
        `;
        els.subQueries.appendChild(div);
    });
}

function renderAnswer(data) {
    // Render Evidence
    if (data.evidence_summary && data.evidence_summary.length > 0) {
        els.evidence.innerHTML = '';
        data.evidence_summary.forEach(sq => {
            sq.top_chunks.forEach(chunk => {
                const div = document.createElement('div');
                div.className = 'ev-card';
                div.innerHTML = `
                    <div class="ev-header">
                        <span class="badge">[${chunk.doc_id} ${chunk.section}]</span>
                        <span class="ev-score">Score: ${chunk.fused_score.toFixed(3)}</span>
                    </div>
                    <div class="ev-text">${chunk.text_preview}...</div>
                `;
                els.evidence.appendChild(div);
            });
        });
    }

    // Render Answer
    els.ansVersion.textContent = data.version;
    els.answer.innerHTML = '';
    
    if (data.is_no_retrieval && data.reformulated_text) {
        // Just text, no specific claims (reformat)
        const p = document.createElement('div');
        p.className = 'claim changed'; // animate it
        p.innerHTML = data.reformulated_text.replace(/\[(Doc_\d+\s*§\d+)\]/g, '<span class="citation">[$1]</span>');
        els.answer.appendChild(p);
        return;
    }

    if (data.claims && data.claims.length > 0) {
        data.claims.forEach(claim => {
            const div = document.createElement('div');
            // If the claim's version matches the current answer version and it's > 1, it changed
            const isChanged = (data.version > 1 && claim.version_introduced === data.version);
            div.className = `claim ${isChanged ? 'changed' : ''}`;
            
            // Format citations
            let text = claim.text;
            text = text.replace(/\[(Doc_\d+\s*§\d+)\]/g, '<span class="citation">[$1]</span>');
            
            div.innerHTML = text;
            els.answer.appendChild(div);
        });
    }
    
    if (data.uncertainty && data.uncertainty.length > 0) {
        data.uncertainty.forEach(unc => {
            const div = document.createElement('div');
            div.className = 'uncertainty-flag';
            div.innerHTML = `<strong>⚠ Insufficient Evidence</strong><br>${unc}`;
            els.answer.appendChild(div);
        });
    }
    
    // If completely empty after rendering (no claims, no uncertainty), show a fallback
    if (els.answer.children.length === 0) {
        const div = document.createElement('div');
        div.className = 'uncertainty-flag';
        div.innerHTML = '<strong>⚠ No verifiable answer</strong><br>The retrieved evidence could not be verified against the corpus for this query.';
        els.answer.appendChild(div);
    }
    
    previousAnswerVersion = data.version;
}

// ─── Scenario Runner ──────────────────────────────────────────────────────

async function runScenario(scenarioId) {
    if (!currentSessionId) await createNewSession();
    
    const chunks = SCENARIOS[scenarioId];
    if (!chunks) return;
    
    // Clear input box
    els.chunkInput.value = "";
    
    // Stream chunks one by one with a delay to simulate typing/speech
    let i = 0;
    if (streamInterval) clearInterval(streamInterval);
    
    streamInterval = setInterval(async () => {
        if (i < chunks.length) {
            await sendChunk(chunks[i]);
            i++;
        } else {
            clearInterval(streamInterval);
            // Auto end-turn when scenario chunks are done
            await endTurn();
        }
    }, 1000); // 1 second between chunks
}

// ─── Event Listeners ──────────────────────────────────────────────────────

els.btnNewSession.addEventListener('click', createNewSession);

els.btnLoadScenario.addEventListener('click', () => {
    // If it's LD-02, we don't clear the session, we just send the new chunk
    const selected = els.scenarioSelect.value;
    if (selected === "LD-02" || selected === "NR-01") {
        if (!currentSessionId) {
            alert("Please run an initial scenario (like MI-01 or LD-01) first to establish session context.");
            return;
        }
        runScenario(selected);
    } else {
        createNewSession().then(() => runScenario(selected));
    }
});

els.btnSendChunk.addEventListener('click', () => {
    const text = els.chunkInput.value.trim();
    if (text) {
        sendChunk(text);
        els.chunkInput.value = '';
    }
});

els.chunkInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        const text = els.chunkInput.value.trim();
        if (text) {
            sendChunk(text);
            els.chunkInput.value = '';
        }
    }
});

els.btnEndTurn.addEventListener('click', endTurn);

els.btnClearLog.addEventListener('click', () => {
    els.telemetry.innerHTML = '';
});

// Auto-start on load
window.addEventListener('DOMContentLoaded', () => {
    createNewSession();
});
