/* ==========================================================================
   Nautilus House — single-page detective UI (vanilla JS, no build step).

   Talks to the FastAPI backend (same origin, mounted at /):
     REST : POST /game/start, GET /game/state, POST /game/action, GET /gm/state
     WS   : /game/stream   (interrogate/scene stream with a typing effect;
                             examine/accuse return a single "result" message)

   All game logic lives server-side; this file only renders state and relays
   player actions.
   ========================================================================== */
"use strict";

/* --------------------------------------------------------------------------
   Module state
   -------------------------------------------------------------------------- */
const State = {
  started: false,
  turn: 0,
  roster: [],            // [{id, name, surface}]
  evidence: [],          // [{clue_id, category, discovered, statement}]
  selected: new Set(),   // selected character ids
  sceneMode: false,      // multi-select scene mode
  gmOpen: false,
  ws: null,
  wsReady: false,
  streaming: false,      // an interrogate/scene reply is in flight
  gmTimer: null,
};

// Deterministic avatar palette (index by roster position).
const AVATAR_COLORS = [
  ["#c9a24b", "#8a7233"],
  ["#4a7ba6", "#2c4d68"],
  ["#3fb6a8", "#237a70"],
  ["#b7657f", "#7a3f52"],
  ["#7c8fb0", "#4c5a75"],
  ["#a67c52", "#6b4e33"],
];

/* --------------------------------------------------------------------------
   Tiny DOM helpers
   -------------------------------------------------------------------------- */
const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

function initials(name) {
  const parts = String(name || "?").trim().split(/\s+/);
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

/* --------------------------------------------------------------------------
   Banner (recoverable errors / info)
   -------------------------------------------------------------------------- */
function showBanner(text, kind = "error") {
  const b = $("#banner");
  $("#banner-text").textContent = text;
  b.classList.remove("hidden", "info");
  if (kind === "info") b.classList.add("info");
}
function hideBanner() {
  $("#banner").classList.add("hidden");
}

/* --------------------------------------------------------------------------
   REST calls
   -------------------------------------------------------------------------- */
async function apiStart() {
  const res = await fetch("/game/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  if (!res.ok) throw new Error(`start failed: ${res.status}`);
  return res.json();
}

async function apiState() {
  const res = await fetch("/game/state");
  if (!res.ok) throw new Error(`state failed: ${res.status}`);
  return res.json();
}

async function apiGmState() {
  const res = await fetch("/gm/state");
  if (!res.ok) throw new Error(`gm state failed: ${res.status}`);
  return res.json();
}

/* --------------------------------------------------------------------------
   WebSocket
   -------------------------------------------------------------------------- */
function connectWs() {
  if (State.ws && (State.wsReady || State.ws.readyState === WebSocket.CONNECTING)) {
    return;
  }
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/game/stream`);
  State.ws = ws;

  ws.addEventListener("open", () => {
    State.wsReady = true;
  });
  ws.addEventListener("close", () => {
    State.wsReady = false;
    State.streaming = false;
    finalizeActiveBubble();
  });
  ws.addEventListener("error", () => {
    // Errors surface via the close handler; keep the socket recoverable.
  });
  ws.addEventListener("message", (evt) => {
    let msg;
    try {
      msg = JSON.parse(evt.data);
    } catch {
      return;
    }
    handleWsMessage(msg);
  });
}

function wsSend(action) {
  if (!State.ws || State.ws.readyState !== WebSocket.OPEN) {
    showBanner("Connection not ready — retrying. Try the action again in a moment.");
    connectWs();
    return false;
  }
  State.ws.send(JSON.stringify(action));
  return true;
}

/* --------------------------------------------------------------------------
   Transcript rendering + typing effect
   -------------------------------------------------------------------------- */
let activeBubble = null;      // { el, textEl, speaker, accum }

function clearTranscript() {
  $("#transcript").innerHTML = "";
}

function scrollTranscript() {
  const t = $("#transcript");
  t.scrollTop = t.scrollHeight;
}

function nameFor(id) {
  if (id === "player") return "You";
  const c = State.roster.find((r) => r.id === id);
  return c ? c.name : id;
}

function addPlayerBubble(text, actionType) {
  const b = el("div", "bubble player");
  b.appendChild(el("span", "speaker", "You"));
  let body = text || "";
  if (actionType === "present_evidence") body = `[Presents evidence] ${body}`;
  else if (actionType === "accuse") body = `[Accuses] ${body}`;
  b.appendChild(el("div", "body", body));
  $("#transcript").appendChild(b);
  scrollTranscript();
}

function addSystemBubble(text) {
  const b = el("div", "bubble system", text);
  $("#transcript").appendChild(b);
  scrollTranscript();
}

/** Start (or continue) a character bubble for a streamed speaker. */
function ensureCharacterBubble(speaker) {
  if (activeBubble && activeBubble.speaker === speaker) return activeBubble;
  finalizeActiveBubble();
  const b = el("div", "bubble character");
  b.appendChild(el("span", "speaker", nameFor(speaker)));
  const textEl = el("div", "body caret");
  b.appendChild(textEl);
  $("#transcript").appendChild(b);
  activeBubble = { el: b, textEl, speaker, accum: "" };
  scrollTranscript();
  return activeBubble;
}

function appendDelta(speaker, delta) {
  const bubble = ensureCharacterBubble(speaker);
  bubble.accum += delta;
  bubble.textEl.textContent = bubble.accum;
  scrollTranscript();
}

/** Remove the typing caret without extra decoration. */
function finalizeActiveBubble() {
  if (activeBubble) {
    activeBubble.textEl.classList.remove("caret");
    activeBubble = null;
  }
}

/** Finalize a bubble and attach reveal/violation/trust metadata (done msg). */
function finalizeDone(msg) {
  const bubble = ensureCharacterBubble(msg.speaker);
  if (typeof msg.text === "string" && msg.text.length) {
    bubble.textEl.textContent = msg.text; // authoritative full text
  }
  bubble.textEl.classList.remove("caret");

  if (Array.isArray(msg.newly_revealed) && msg.newly_revealed.length) {
    const summaries = msg.newly_revealed
      .map((r) => r.summary || r.fact_id)
      .filter(Boolean);
    if (summaries.length) {
      bubble.el.appendChild(
        el("div", "reveal-tag", `revealed: ${summaries.join("; ")}`)
      );
    }
  }
  if (Array.isArray(msg.violations) && msg.violations.length) {
    bubble.el.appendChild(
      el("div", "violation-tag", `⚠ leak-guard: ${msg.violations.length} flag(s)`)
    );
  }
  if (msg.trust_band) {
    bubble.el.appendChild(
      el("div", "trust-chip", `stance toward you: ${msg.trust_band}`)
    );
  }
  activeBubble = null;
  scrollTranscript();
}

/* --------------------------------------------------------------------------
   WS message dispatch
   -------------------------------------------------------------------------- */
function handleWsMessage(msg) {
  switch (msg.type) {
    case "chunk":
      appendDelta(msg.speaker, msg.delta || "");
      break;

    case "done":
      finalizeDone(msg);
      if (typeof msg.turn === "number") setTurn(msg.turn);
      // interrogate finishes on its "done"; scene finishes on scene_complete.
      afterActionRefresh();
      State.streaming = false;
      updateSendEnabled();
      break;

    case "scene_complete":
      finalizeActiveBubble();
      State.streaming = false;
      updateSendEnabled();
      afterActionRefresh();
      break;

    case "result":
      handleResult(msg);
      if (typeof msg.turn === "number") setTurn(msg.turn);
      State.streaming = false;
      updateSendEnabled();
      afterActionRefresh();
      break;

    case "error":
      handleWsError(msg.detail || "Unknown error.");
      State.streaming = false;
      updateSendEnabled();
      break;

    default:
      break;
  }
}

function handleWsError(detail) {
  finalizeActiveBubble();
  const lower = detail.toLowerCase();
  if (lower.includes("openrouter_api_key") || lower.includes("api key")) {
    showBanner(
      "No LLM key configured: set OPENROUTER_API_KEY in your .env (see " +
        ".env.example) and restart the server to interrogate suspects.",
      "info"
    );
  } else {
    showBanner(detail);
  }
}

function handleResult(msg) {
  // examine result -> has clue_id; accuse result -> has "win".
  if (Object.prototype.hasOwnProperty.call(msg, "win")) {
    renderAccusationResult(msg);
  } else if (msg.clue_id) {
    const label = msg.newly_discovered ? "Examined" : "Re-examined";
    const stmt = msg.statement ? `: ${msg.statement}` : "";
    addSystemBubble(`${label} ${msg.clue_id}${stmt}`);
  }
}

function renderAccusationResult(msg) {
  const sol = msg.solution || {};
  const cls = msg.win ? "result-win" : "result-lose";
  const b = el("div", `bubble ${cls}`);
  const headline = msg.win
    ? "🕯 Case solved — you named the killer correctly."
    : "✦ Wrong. The killer walks free as the boat arrives.";
  b.appendChild(el("div", "headline", headline));
  const detail = el("div", "detail");
  const bits = [];
  bits.push(`killer: ${msg.correct_killer ? "correct" : "wrong"}`);
  if (msg.correct_room !== null && msg.correct_room !== undefined)
    bits.push(`room: ${msg.correct_room ? "correct" : "wrong"}`);
  if (msg.correct_weapon !== null && msg.correct_weapon !== undefined)
    bits.push(`weapon: ${msg.correct_weapon ? "correct" : "wrong"}`);
  detail.textContent = bits.join(" · ");
  b.appendChild(detail);
  if (!msg.win && sol.killer) {
    b.appendChild(
      el(
        "div",
        "detail",
        `Solution — killer: ${nameFor(sol.killer)}, room: ${sol.room}, weapon: ${sol.weapon}`
      )
    );
  }
  $("#transcript").appendChild(b);
  scrollTranscript();
}

/* --------------------------------------------------------------------------
   State refresh
   -------------------------------------------------------------------------- */
function setTurn(t) {
  State.turn = t;
  $("#turn-counter").textContent = t;
}

async function afterActionRefresh() {
  try {
    const s = await apiState();
    setTurn(s.turn);
    State.evidence = s.evidence_board || [];
    renderEvidence();
  } catch (err) {
    // Non-fatal; leave last-known state.
  }
  if (State.gmOpen) refreshGm();
}

/* --------------------------------------------------------------------------
   Roster
   -------------------------------------------------------------------------- */
function renderRoster() {
  const wrap = $("#roster");
  wrap.innerHTML = "";
  State.roster.forEach((c, i) => {
    const card = el("button", "suspect-card");
    card.type = "button";
    card.dataset.id = c.id;
    if (State.selected.has(c.id)) card.classList.add("selected");

    const [c1, c2] = AVATAR_COLORS[i % AVATAR_COLORS.length];
    const av = el("div", "avatar", initials(c.name));
    av.style.background = `radial-gradient(circle at 30% 25%, ${c1}, ${c2})`;

    const who = el("div", "who");
    who.appendChild(el("div", "name", c.name));
    who.appendChild(el("div", "surface", c.surface || ""));

    card.appendChild(av);
    card.appendChild(who);
    card.appendChild(el("span", "select-badge", "● selected"));

    card.addEventListener("click", (e) => onSelectSuspect(c.id, e.shiftKey));
    wrap.appendChild(card);
  });
  refreshCardSelectionClasses();
}

function refreshCardSelectionClasses() {
  document.querySelectorAll(".suspect-card").forEach((card) => {
    card.classList.toggle("selected", State.selected.has(card.dataset.id));
  });
}

function onSelectSuspect(id, shiftKey) {
  if (!State.started) return;
  const multi = State.sceneMode || shiftKey;
  if (multi) {
    if (State.selected.has(id)) State.selected.delete(id);
    else State.selected.add(id);
  } else {
    State.selected.clear();
    State.selected.add(id);
  }
  refreshCardSelectionClasses();
  updateChatHeader();
  updateSendEnabled();
}

function updateChatHeader() {
  const label = $("#chat-target-label");
  const sel = [...State.selected];
  if (sel.length === 0) label.textContent = "Interrogation";
  else if (sel.length === 1) label.textContent = `Interrogating ${nameFor(sel[0])}`;
  else label.textContent = `Scene: ${sel.map(nameFor).join(", ")}`;
}

/* --------------------------------------------------------------------------
   Evidence board
   -------------------------------------------------------------------------- */
function prettyClueName(clueId) {
  return clueId
    .replace(/^clue_/, "")
    .replace(/^anchor_/, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function renderEvidence() {
  const wrap = $("#evidence-board");
  wrap.innerHTML = "";
  if (!State.evidence.length) {
    wrap.appendChild(el("div", "col-hint", "No clues yet — start a game."));
  }
  State.evidence.forEach((clue) => {
    const card = el("div", "clue" + (clue.discovered ? "" : " locked"));
    const head = el("div", "clue-head");
    head.appendChild(el("div", "clue-name", prettyClueName(clue.clue_id)));
    const cat = el("div", "clue-cat" + (clue.category === "anchor" ? " anchor" : ""),
      clue.category || "clue");
    head.appendChild(cat);
    card.appendChild(head);

    if (clue.discovered) {
      card.appendChild(el("div", "clue-statement", clue.statement || ""));
    } else {
      card.appendChild(el("div", "clue-locked-label", "🔒 Undiscovered"));
      const btn = el("button", "btn examine-btn", "Examine");
      btn.type = "button";
      btn.addEventListener("click", () => examineClue(clue.clue_id));
      card.appendChild(btn);
    }
    wrap.appendChild(card);
  });
  renderEvidencePicker();
}

/** Populate the "present evidence" dropdown with discovered clues. */
function renderEvidencePicker() {
  const sel = $("#evidence-picker");
  const prev = sel.value;
  sel.innerHTML = "";
  const discovered = State.evidence.filter((c) => c.discovered);
  if (!discovered.length) {
    sel.appendChild(new Option("(no discovered clues)", ""));
  } else {
    discovered.forEach((c) =>
      sel.appendChild(new Option(prettyClueName(c.clue_id), c.clue_id))
    );
  }
  if (prev && discovered.some((c) => c.clue_id === prev)) sel.value = prev;
}

/* --------------------------------------------------------------------------
   Accusation dropdown
   -------------------------------------------------------------------------- */
function renderAccuseKiller() {
  const sel = $("#accuse-killer");
  sel.innerHTML = "";
  sel.appendChild(new Option("— choose —", ""));
  State.roster.forEach((c) => sel.appendChild(new Option(c.name, c.id)));
}

/* --------------------------------------------------------------------------
   Actions
   -------------------------------------------------------------------------- */
function updateSendEnabled() {
  const hasSel = State.selected.size > 0;
  const busy = State.streaming;
  $("#send-btn").disabled = !State.started || !hasSel || State.selected.size !== 1 || busy;
  $("#scene-btn").disabled = !State.started || State.selected.size < 2 || busy;
  $("#accuse-btn").disabled = !State.started || busy;
}

function sendInterrogate() {
  const id = [...State.selected][0];
  if (!id) return;
  const actionType = $("#action-type").value;
  const utterance = $("#utterance").value.trim();
  if (!utterance) {
    showBanner("Type something to say first.", "info");
    return;
  }

  const action = {
    type: "interrogate",
    character_id: id,
    utterance,
    action_type: actionType,
  };
  if (actionType === "present_evidence") {
    const clueId = $("#evidence-picker").value;
    if (clueId) action.clue_id = clueId;
  }

  addPlayerBubble(utterance, actionType);
  $("#utterance").value = "";
  State.streaming = true;
  updateSendEnabled();
  wsSend(action);
}

function sendScene() {
  const ids = [...State.selected];
  if (ids.length < 2) return;
  const utterance = $("#utterance").value.trim();
  if (!utterance) {
    showBanner("Type something to say to the room first.", "info");
    return;
  }
  addPlayerBubble(utterance, "ask");
  addSystemBubble(`— Scene convened with ${ids.map(nameFor).join(", ")} —`);
  $("#utterance").value = "";
  State.streaming = true;
  updateSendEnabled();
  wsSend({ type: "scene", character_ids: ids, utterance });
}

function examineClue(clueId) {
  if (!State.started) return;
  wsSend({ type: "examine", clue_id: clueId });
}

function makeAccusation() {
  const killer = $("#accuse-killer").value;
  if (!killer) {
    showBanner("Pick who you're accusing.", "info");
    return;
  }
  const room = $("#accuse-room").value.trim();
  const weapon = $("#accuse-weapon").value.trim();
  const action = { type: "accuse", character_id: killer };
  if (room) action.room = room;
  if (weapon) action.weapon = weapon;
  wsSend(action);
}

/* --------------------------------------------------------------------------
   GM / Judge panel
   -------------------------------------------------------------------------- */
function toggleGm(force) {
  State.gmOpen = force !== undefined ? force : !State.gmOpen;
  $("#gm-panel").classList.toggle("hidden", !State.gmOpen);
  $("#gm-scrim").classList.toggle("hidden", !State.gmOpen);
  $("#gm-panel").setAttribute("aria-hidden", String(!State.gmOpen));
  $("#gm-toggle-btn").setAttribute("aria-pressed", String(State.gmOpen));
  if (State.gmOpen && State.started) refreshGm();
}

async function refreshGm() {
  if (!State.started) return;
  let gm;
  try {
    gm = await apiGmState();
  } catch {
    return;
  }
  renderLeakDash(gm);
  renderDisclosure(gm.disclosure || {});
  renderTrustGraph(gm.trust_graph || {});
}

function renderLeakDash(gm) {
  const violations = gm.leak_violations || [];
  $("#leak-turns").textContent = gm.turn ?? 0;
  $("#leak-count").textContent = violations.length;
  const ind = $("#leak-indicator");
  ind.classList.toggle("clean", violations.length === 0);
  ind.classList.toggle("dirty", violations.length > 0);

  const list = $("#leak-list");
  list.innerHTML = "";
  violations.forEach((v) => {
    const li = el(
      "li",
      null,
      `Turn ${v.turn} — ${nameFor(v.character_id)}: ${
        Array.isArray(v.violations) ? v.violations.join(", ") : v.violations
      }`
    );
    list.appendChild(li);
  });
}

function renderDisclosure(disclosure) {
  const wrap = $("#disclosure");
  wrap.innerHTML = "";
  Object.keys(disclosure).forEach((cid) => {
    const d = disclosure[cid];
    const box = el("div", "disc-char");
    box.appendChild(el("div", "disc-name", nameFor(cid)));
    const tiers = el("div", "disc-tiers");
    const tierMap = d.tiers || {};
    Object.keys(tierMap)
      .sort()
      .forEach((tk) => {
        const t = tierMap[tk];
        const chip = el("div", `tier ${t.state}`);
        chip.appendChild(el("span", "dot"));
        chip.appendChild(document.createTextNode(`T${tk} ${t.state}`));
        chip.appendChild(el("span", "qs", ` (${t.questions_seen}q)`));
        chip.title = t.fact_id || "";
        tiers.appendChild(chip);
      });
    box.appendChild(tiers);
    wrap.appendChild(box);
  });
}

/* --------------------------------------------------------------------------
   Trust graph — 7 nodes on a circle, directed weighted edges on an SVG
   -------------------------------------------------------------------------- */
const SVG_NS = "http://www.w3.org/2000/svg";

function renderTrustGraph(trust) {
  const svg = $("#trust-graph");
  svg.innerHTML = "";

  // Node set = union of all from/to ids present in the map (6 chars + player).
  const nodeSet = new Set();
  Object.keys(trust).forEach((from) => {
    nodeSet.add(from);
    Object.keys(trust[from] || {}).forEach((to) => nodeSet.add(to));
  });
  // Stable ordering: roster order first, then player, then any extras.
  const ordered = [];
  State.roster.forEach((r) => nodeSet.has(r.id) && ordered.push(r.id));
  if (nodeSet.has("player")) ordered.push("player");
  [...nodeSet].forEach((n) => !ordered.includes(n) && ordered.push(n));

  const size = 320;
  const cx = size / 2;
  const cy = size / 2;
  const radius = 118;
  const nodeR = 20;
  const pos = {};
  ordered.forEach((id, i) => {
    const angle = -Math.PI / 2 + (i / ordered.length) * 2 * Math.PI;
    pos[id] = { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
  });

  // Arrow marker defs (one green, one red).
  const defs = document.createElementNS(SVG_NS, "defs");
  defs.appendChild(makeMarker("arrow-pos", "#4bb06a"));
  defs.appendChild(makeMarker("arrow-neg", "#c8583f"));
  svg.appendChild(defs);

  // Edges (draw before nodes so nodes sit on top).
  const EDGE_MIN = 0.08; // hide near-zero edges to reduce clutter
  ordered.forEach((from) => {
    const row = trust[from] || {};
    ordered.forEach((to) => {
      if (from === to) return;
      const score = row[to];
      if (typeof score !== "number" || Math.abs(score) < EDGE_MIN) return;
      drawEdge(svg, pos[from], pos[to], score, nodeR);
    });
  });

  // Nodes.
  ordered.forEach((id, i) => {
    const p = pos[id];
    const g = document.createElementNS(SVG_NS, "g");

    const circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("cx", p.x);
    circle.setAttribute("cy", p.y);
    circle.setAttribute("r", nodeR);
    circle.setAttribute("class", "node-circle" + (id === "player" ? " player" : ""));
    if (id === "player") {
      circle.setAttribute("fill", "#1c2b36");
    } else {
      const idx = State.roster.findIndex((r) => r.id === id);
      const [c1] = AVATAR_COLORS[(idx >= 0 ? idx : i) % AVATAR_COLORS.length];
      circle.setAttribute("fill", c1);
    }
    g.appendChild(circle);

    const label = document.createElementNS(SVG_NS, "text");
    label.setAttribute("x", p.x);
    label.setAttribute("y", p.y + 3);
    const short = id === "player" ? "YOU" : initials(nameFor(id));
    label.textContent = short;
    g.appendChild(label);

    const full = document.createElementNS(SVG_NS, "title");
    full.textContent = nameFor(id);
    g.appendChild(full);

    svg.appendChild(g);
  });
}

function makeMarker(id, color) {
  const m = document.createElementNS(SVG_NS, "marker");
  m.setAttribute("id", id);
  m.setAttribute("viewBox", "0 0 10 10");
  m.setAttribute("refX", "9");
  m.setAttribute("refY", "5");
  m.setAttribute("markerWidth", "6");
  m.setAttribute("markerHeight", "6");
  m.setAttribute("orient", "auto-start-reverse");
  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
  path.setAttribute("fill", color);
  m.appendChild(path);
  return m;
}

function drawEdge(svg, a, b, score, nodeR) {
  // Shorten the segment so it starts/ends at node borders (not centers), and
  // offset perpendicular slightly so A->B and B->A don't overlap.
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  const px = -uy; // perpendicular
  const py = ux;
  const off = 5;

  const x1 = a.x + ux * nodeR + px * off;
  const y1 = a.y + uy * nodeR + py * off;
  const x2 = b.x - ux * (nodeR + 6) + px * off;
  const y2 = b.y - uy * (nodeR + 6) + py * off;

  const pos = score >= 0;
  const color = pos ? "#4bb06a" : "#c8583f";
  const width = 1 + Math.min(Math.abs(score), 1) * 4;

  const line = document.createElementNS(SVG_NS, "line");
  line.setAttribute("x1", x1);
  line.setAttribute("y1", y1);
  line.setAttribute("x2", x2);
  line.setAttribute("y2", y2);
  line.setAttribute("stroke", color);
  line.setAttribute("stroke-width", width.toFixed(2));
  line.setAttribute("stroke-linecap", "round");
  line.setAttribute("opacity", (0.35 + Math.min(Math.abs(score), 1) * 0.5).toFixed(2));
  line.setAttribute("marker-end", pos ? "url(#arrow-pos)" : "url(#arrow-neg)");
  const title = document.createElementNS(SVG_NS, "title");
  title.textContent = score.toFixed(2);
  line.appendChild(title);
  svg.appendChild(line);
}

/* --------------------------------------------------------------------------
   New game
   -------------------------------------------------------------------------- */
async function newGame() {
  hideBanner();
  $("#new-game-btn").disabled = true;
  try {
    const start = await apiStart();
    State.started = true;
    State.turn = start.turn || 0;
    State.roster = start.roster || [];
    State.evidence = start.evidence_board || [];
    State.selected.clear();

    setTurn(State.turn);
    renderRoster();
    renderEvidence();
    renderAccuseKiller();
    updateChatHeader();
    updateSendEnabled();

    clearTranscript();
    addSystemBubble(
      "A new night at Nautilus House. Six suspects, one storm, one killer. " +
        "Select a suspect to begin."
    );

    connectWs();
    if (State.gmOpen) refreshGm();
  } catch (err) {
    showBanner(`Could not start a new game: ${err.message}`);
  } finally {
    $("#new-game-btn").disabled = false;
  }
}

/* --------------------------------------------------------------------------
   Wire up controls
   -------------------------------------------------------------------------- */
function wireControls() {
  $("#new-game-btn").addEventListener("click", newGame);
  $("#gm-toggle-btn").addEventListener("click", () => toggleGm());
  $("#gm-close-btn").addEventListener("click", () => toggleGm(false));
  $("#gm-scrim").addEventListener("click", () => toggleGm(false));
  $("#banner-close").addEventListener("click", hideBanner);

  $("#send-btn").addEventListener("click", sendInterrogate);
  $("#scene-btn").addEventListener("click", sendScene);
  $("#accuse-btn").addEventListener("click", makeAccusation);

  $("#scene-mode-toggle").addEventListener("change", (e) => {
    State.sceneMode = e.target.checked;
    if (!State.sceneMode && State.selected.size > 1) {
      // Collapse to a single selection when leaving scene mode.
      const first = [...State.selected][0];
      State.selected.clear();
      if (first) State.selected.add(first);
      refreshCardSelectionClasses();
      updateChatHeader();
    }
    updateSendEnabled();
  });

  $("#action-type").addEventListener("change", (e) => {
    const isEvidence = e.target.value === "present_evidence";
    $("#evidence-picker-wrap").hidden = !isEvidence;
  });

  // Enter to send (Shift+Enter = newline).
  $("#utterance").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (State.selected.size >= 2) sendScene();
      else sendInterrogate();
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && State.gmOpen) toggleGm(false);
  });
}

/* --------------------------------------------------------------------------
   Boot
   -------------------------------------------------------------------------- */
function boot() {
  wireControls();
  updateSendEnabled();
  // Try to attach to an existing session (e.g. page reload mid-game).
  apiState()
    .then((s) => {
      State.started = true;
      State.turn = s.turn || 0;
      State.roster = s.roster || [];
      State.evidence = s.evidence_board || [];
      setTurn(State.turn);
      renderRoster();
      renderEvidence();
      renderAccuseKiller();
      updateChatHeader();
      updateSendEnabled();
      clearTranscript();
      addSystemBubble("Resumed the current session. Select a suspect to continue.");
      connectWs();
    })
    .catch(() => {
      // 409 (no game yet) is expected on a fresh load; wait for New Game.
    });
}

document.addEventListener("DOMContentLoaded", boot);
