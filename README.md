# Nautilus House — Multi-Agent Detective Game

A turn-based web mystery where six LLM-backed suspects, each with **private,
partitioned memory** and **per-relationship trust**, are interrogated the night
after a murder at Nautilus House. Every character keeps its own secrets, reacts
to what it hears other characters say, and — crucially — **never leaks facts its
persona wouldn't know**. That last guarantee isn't left to prompt discipline: a
physical content partition plus a runtime, disclosure-aware **leak guard** back
it up, and an automated leak-probe dashboard proves "0 violations across N turns"
for judges. Play the detective in the browser; open the GM / Judge panel to watch
disclosure state, the live trust graph, and the leak dashboard update in real
time.

## The case (spoiler-light)

Ignatius Vance is found dead in the Library. Six suspects are stranded by a storm
for nine hours. The truth: the body was killed elsewhere and *moved*, the murder
weapon (a fossil Ammonite) is hidden on the island, and one suspect is protecting
another for entirely the wrong reason. The full solution lives only in
`content/ground_truth.json`, which agent code never reads.

## Architecture

The design is a strict split between **agent-facing content** (safe to put in an
LLM prompt) and **GM-only content** (the solution), enforced by *which loader
function is called*, not by prompt wording.

| Layer | File(s) | Responsibility |
|-------|---------|----------------|
| **Content partition** | `content/`, `backend/content_loader.py` | Public canon + per-character dossiers (agent-facing) vs. `ground_truth.json` / `leak_probes.json` (GM-only `gm_*` loaders). The `gm_*` functions are never imported by any prompt-building code. |
| **Per-character memory** | `backend/character_agent.py` | One `CharacterAgent` per suspect: static persona, tiered private knowledge, own episodic history, and `heard_from_others`. Its prompt assembler draws **only** from its own dossier + public canon + the trust band + currently-unlocked facts — so no foreign or solution string is ever in context to leak. |
| **Disclosure ladder** | `backend/disclosure.py` | Tracks each secret tier as `sealed → pressured_once → revealed` from four pressure signals (evidence, social, trade, value/goal-stack inversion). Decides *when* a secret may be revealed; it never writes the line. |
| **Trust graph** | `backend/trust_graph.py` | Directed weighted trust over 6 characters + the player, seeded from the canon relationships, discretized into bands (`hostile…close`) and updated per turn. Trust lowers/raises disclosure thresholds and colors reactions. |
| **Leak guard** | `backend/leak_guard.py` | Runtime, **disclosure-aware** canary scanner. Flags a protected string only if it appears *and* the fact it guards is not yet legitimately `revealed`. Defense-in-depth behind the partition. |
| **Orchestrator** | `backend/orchestrator.py`, `backend/game_state.py` | The single turn-loop seam: `interrogate`, multi-character `scene` (bounded agent-to-agent chain), `examine`, `set_flag`, `accuse_solution`, `gm_state`. Wires disclosure + trust + broadcast + leak scan per utterance and resolves the win against ground truth. |
| **LLM client** | `backend/llm_client.py` | Thin OpenRouter (OpenAI-compatible) wrapper. Importing it never crashes without a key; live calls raise `MissingAPIKeyError`. |
| **Web app** | `backend/app.py`, `frontend/` | FastAPI REST + WebSocket over the orchestrator, serving a plain HTML/CSS/JS UI: roster, chat, evidence board, and a toggleable GM/Judge panel (disclosure, trust graph, leak dashboard). |

## Setup

Requires Python 3.11+.

```bash
pip install -r requirements.txt
cp .env.example .env          # Windows: copy .env.example .env
```

Then edit `.env` and set your OpenRouter key (get one at
<https://openrouter.ai/keys>):

```
OPENROUTER_API_KEY=sk-or-v1-...
# optional: OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
```

The key is **only** needed for live character replies (`interrogate` / `scene`).
Everything else — the guided demo (offline mode), the leak tests, `examine`,
`accuse`, and the whole GM/judge view — runs with no key at all.

## Run the game

```bash
python scripts/run_server.py          # or: uvicorn backend.app:app --port 8000
```

Then open <http://127.0.0.1:8000/> in your browser. Interrogate suspects, examine
clues to build the evidence board, and make a final accusation
(who / room / weapon) to solve the case.

**GM / Judge panel:** click the **"GM / Judge Panel"** button in the top bar (or
hit `GET /gm/state`). This is the "human teammate" view — it shows each
character's per-tier disclosure state, the live trust graph, discovered evidence,
and the running leak-violation log. It's the only place the internals are
exposed; the solution file is still never sent to any agent.

Without an `OPENROUTER_API_KEY`, the server still boots and serves the UI;
`interrogate`/`scene` return a clean `503` explaining the key is missing, while
`examine`/`accuse`/`GET /gm/state` keep working.

## Run the guided demo (no key required)

A narrated, scripted playthrough that maps to the "show the judges" moment —
a killer holding the murder room under a direct probe, a tiered secret opening
under evidence while a deeper one stays sealed, a goal-stack inversion that
unseals a confession, the winning accusation, and a final **ground-truth +
leak-dashboard** summary:

```bash
python scripts/demo_playthrough.py
```

It auto-detects its mode: **live** if `OPENROUTER_API_KEY` is set (real replies),
otherwise **offline/mock** (a deterministic, leak-safe in-character stub, zero
external calls). The mode is printed in the banner.

## Run the leak tests

The judge-facing leak-probe dashboard — replays every probe in
`content/leak_probes.json` and prints a pass/fail table:

```bash
python scripts/run_leak_tests.py
```

With a key it runs **live** (scans real replies); without one it runs in
**structural** mode (proves the prompt-level partition holds). A healthy build
reports `0 violations` and exits `0`.

## Other checks

```bash
python -m backend.content_loader     # content loads + partition wiring
python -m backend.character_agent    # prompt-assembly / no-leak assertions
python -m backend.disclosure         # tiered unlock behavior
python -m backend.trust_graph        # trust seeding + band translation
python -m backend.orchestrator       # full offline turn-loop smoke test
python -m backend.leak_guard         # disclosure-aware scanner self-check
python scripts/test_app.py           # FastAPI TestClient smoke test
```

## Repo layout

```
content/         public_canon.json, dossiers/*.json, clues.json,
                 ground_truth.json, leak_probes.json, rooms.json
backend/         content_loader, character_agent, llm_client, disclosure,
                 trust_graph, game_state, orchestrator, leak_guard, app
frontend/        index.html, main.js, styles.css
scripts/         run_server, run_leak_tests, test_app, demo_playthrough,
                 generate_rooms
```
