"""FastAPI backend for the Nautilus House game (plan section 7).

This module is a THIN wrapper around the already-built
:class:`backend.orchestrator.Orchestrator`. It owns no game logic of its own --
it exposes the orchestrator's stable method surface over REST + WebSocket and
serves the (separately-built) static frontend.

Design (per plan section 7):
  * A single, in-memory game session. There is exactly one module-level
    ``Orchestrator``; ``POST /game/start`` creates/resets it. This is a
    single-session live demo -- no DB, no per-client isolation.
  * REST endpoints for start / state / action / GM view.
  * A WebSocket (`/game/stream`) that replays orchestrator replies with a
    typing effect: the orchestrator is called normally (non-streaming), then
    the returned ``text`` is chunked in the API layer into ``chunk`` messages
    followed by a ``done`` message. The orchestrator is never modified to add
    streaming -- all chunking lives here.
  * CORS open to all origins (local demo).
  * The ``frontend/`` directory is mounted for static assets; ``GET /`` serves
    ``frontend/index.html`` when present, otherwise a tiny placeholder so the
    server still boots before the frontend phase exists.

Run it with either::

    uvicorn backend.app:app --reload --port 8000
    python -m backend.app
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.llm_client import MissingAPIKeyError
from backend.orchestrator import Orchestrator

# --------------------------------------------------------------------------- #
# Paths / static frontend
# --------------------------------------------------------------------------- #
# frontend/ is created (if missing) at import time so the StaticFiles mount
# never crashes on a fresh checkout -- the frontend phase drops real files in.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_FRONTEND_DIR = _REPO_ROOT / "frontend"
_FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
_INDEX_HTML = _FRONTEND_DIR / "index.html"

# Placeholder served at GET / until the frontend phase adds a real index.html.
_PLACEHOLDER_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Nautilus House &mdash; backend up</title>
    <style>
      body { font-family: system-ui, sans-serif; margin: 3rem auto; max-width: 40rem;
             line-height: 1.5; color: #1a2733; }
      code { background: #eef2f5; padding: 0.1rem 0.35rem; border-radius: 4px; }
      h1 { margin-bottom: 0.25rem; }
    </style>
  </head>
  <body>
    <h1>Nautilus House</h1>
    <p>The FastAPI backend is running. The frontend has not been built yet.</p>
    <p>Try the API:</p>
    <ul>
      <li><code>POST /game/start</code></li>
      <li><code>GET /game/state</code></li>
      <li><code>POST /game/action</code></li>
      <li><code>GET /gm/state</code></li>
      <li><code>WS /game/stream</code></li>
    </ul>
    <p>Interactive docs at <a href="/docs">/docs</a>.</p>
  </body>
</html>
"""

# Word groups sent per typing-effect chunk over the WebSocket.
_WS_CHUNK_WORDS = 3

# --------------------------------------------------------------------------- #
# Single in-memory session
# --------------------------------------------------------------------------- #
# The whole app shares ONE orchestrator; POST /game/start (re)creates it.
_orchestrator: Optional[Orchestrator] = None


def _require_orchestrator() -> Orchestrator:
    """Return the active orchestrator or raise 409 if no game has started."""
    if _orchestrator is None:
        raise HTTPException(
            status_code=409,
            detail="No game in progress. POST /game/start first.",
        )
    return _orchestrator


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class StartRequest(BaseModel):
    """Optional seed so a demo run is reproducible."""

    seed: Optional[int] = None


class ActionRequest(BaseModel):
    """One request model covering every action type.

    ``type`` selects the orchestrator method; the remaining fields are the
    union of every method's parameters (only the relevant ones are used):

      * ``interrogate`` -> ``character_id``, ``utterance``, ``action_type``
      * ``scene``       -> ``character_ids``, ``utterance``
      * ``examine``     -> ``clue_id``
      * ``accuse``      -> ``character_id``, ``room``, ``weapon``
    """

    type: str
    character_id: Optional[str] = None
    character_ids: Optional[list[str]] = None
    utterance: Optional[str] = None
    action_type: Optional[str] = "ask"
    clue_id: Optional[str] = None
    room: Optional[str] = None
    weapon: Optional[str] = None


# --------------------------------------------------------------------------- #
# Action dispatch (shared by REST + WebSocket)
# --------------------------------------------------------------------------- #
_MISSING_KEY_DETAIL = (
    "OPENROUTER_API_KEY is not set, so character replies (interrogate/scene) "
    "cannot be generated. Add OPENROUTER_API_KEY to your .env (see "
    ".env.example) and restart the server."
)


def _dispatch_non_llm(orch: Orchestrator, req: ActionRequest) -> dict:
    """Handle the non-LLM actions (examine/accuse) or raise HTTP errors.

    Returns ``None`` when ``req.type`` is an LLM-backed action (interrogate/
    scene) so the caller can handle streaming vs. blocking separately.
    """
    action = req.type
    try:
        if action == "examine":
            if not req.clue_id:
                raise HTTPException(422, "examine requires 'clue_id'.")
            return orch.examine(req.clue_id)
        if action == "accuse":
            if not req.character_id:
                raise HTTPException(422, "accuse requires 'character_id'.")
            return orch.accuse_solution(
                req.character_id, room=req.room, weapon=req.weapon
            )
    except ValueError as exc:  # unknown clue/character id, etc.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise HTTPException(status_code=400, detail=f"Unknown action type '{action}'.")


def _run_llm_action(orch: Orchestrator, req: ActionRequest) -> Any:
    """Run interrogate/scene, translating orchestrator errors to HTTP errors.

    Raises :class:`MissingAPIKeyError` upward (the REST handler maps it to 503;
    the WS handler maps it to an ``error`` message).
    """
    action = req.type
    try:
        if action == "interrogate":
            if not req.character_id:
                raise HTTPException(422, "interrogate requires 'character_id'.")
            return orch.interrogate(
                req.character_id,
                req.utterance or "",
                action_type=req.action_type or "ask",
            )
        if action == "scene":
            if not req.character_ids:
                raise HTTPException(422, "scene requires 'character_ids'.")
            return orch.scene(req.character_ids, req.utterance or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise HTTPException(status_code=400, detail=f"Unknown action type '{action}'.")


_LLM_ACTIONS = {"interrogate", "scene"}


# --------------------------------------------------------------------------- #
# App + middleware
# --------------------------------------------------------------------------- #
app = FastAPI(title="Nautilus House Mystery", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# REST endpoints
# --------------------------------------------------------------------------- #
@app.post("/game/start")
def game_start(req: StartRequest | None = None) -> dict:
    """Create/reset the single in-memory session and return the start state."""
    global _orchestrator
    seed = req.seed if req else None
    _orchestrator = Orchestrator(seed)
    return _orchestrator.start()


@app.get("/game/state")
def game_state() -> dict:
    """Player-facing snapshot (roster, evidence board, transcript, turn)."""
    return _require_orchestrator().public_state()


@app.post("/game/action")
def game_action(req: ActionRequest) -> Any:
    """Dispatch one player action to the matching orchestrator method.

    * ``examine`` / ``accuse`` run without the LLM.
    * ``interrogate`` / ``scene`` call the LLM; a missing API key becomes a
      clean 503 (not a 500 stack trace).
    """
    orch = _require_orchestrator()
    if req.type in _LLM_ACTIONS:
        try:
            return _run_llm_action(orch, req)
        except MissingAPIKeyError as exc:
            raise HTTPException(status_code=503, detail=_MISSING_KEY_DETAIL) from exc
    return _dispatch_non_llm(orch, req)


@app.get("/gm/state")
def gm_state() -> dict:
    """Judge/GM view: disclosure states, trust graph, evidence, leak log."""
    return _require_orchestrator().gm_state()


# --------------------------------------------------------------------------- #
# WebSocket: typing-effect streaming
# --------------------------------------------------------------------------- #
def _chunk_text(text: str, size: int = _WS_CHUNK_WORDS) -> list[str]:
    """Split ``text`` into small word groups, preserving spacing.

    Each returned piece keeps its trailing whitespace so concatenating all the
    deltas reproduces the original string exactly.
    """
    if not text:
        return []
    # Split into (word + following whitespace) tokens.
    import re

    tokens = re.findall(r"\S+\s*", text)
    if not tokens:
        return [text]
    return ["".join(tokens[i : i + size]) for i in range(0, len(tokens), size)]


async def _stream_reply(ws: WebSocket, reply: dict, delay: float = 0.02) -> None:
    """Stream a single speaker's reply dict as chunk* + done messages."""
    speaker = reply.get("speaker")
    for delta in _chunk_text(reply.get("text", "")):
        await ws.send_json({"type": "chunk", "speaker": speaker, "delta": delta})
        if delay:
            await asyncio.sleep(delay)
    # The full, authoritative reply dict is echoed in the terminal 'done'.
    await ws.send_json({"type": "done", **reply})


@app.websocket("/game/stream")
async def game_stream(ws: WebSocket) -> None:
    """Accept JSON action messages and stream replies with a typing effect.

    Message protocol (server -> client):
      * ``{"type": "chunk", "speaker": <id>, "delta": <text>}`` -- one typed slice.
      * ``{"type": "done", ...<full reply dict>}`` -- terminal per-speaker reply
        (speaker/text/turn/newly_revealed/trust_band/violations).
      * ``{"type": "scene_complete"}`` -- sent after all speakers in a scene.
      * ``{"type": "error", "detail": <msg>}`` -- recoverable error (e.g. no
        API key, bad action); the socket stays open.

    Non-LLM actions (examine/accuse) return a single
    ``{"type": "result", ...}`` message (no typing effect needed).
    """
    await ws.accept()
    try:
        while True:
            payload = await ws.receive_json()
            try:
                req = ActionRequest(**payload)
            except Exception as exc:  # pydantic validation, bad shape, etc.
                await ws.send_json({"type": "error", "detail": f"Bad action: {exc}"})
                continue

            if _orchestrator is None:
                await ws.send_json(
                    {"type": "error", "detail": "No game in progress. Start one first."}
                )
                continue
            orch = _orchestrator

            # --- LLM-backed actions get the typing-effect stream. -----------
            if req.type == "interrogate":
                try:
                    reply = orch.interrogate(
                        req.character_id or "",
                        req.utterance or "",
                        action_type=req.action_type or "ask",
                    )
                except MissingAPIKeyError:
                    await ws.send_json({"type": "error", "detail": _MISSING_KEY_DETAIL})
                    continue
                except ValueError as exc:
                    await ws.send_json({"type": "error", "detail": str(exc)})
                    continue
                await _stream_reply(ws, reply)

            elif req.type == "scene":
                try:
                    replies = orch.scene(req.character_ids or [], req.utterance or "")
                except MissingAPIKeyError:
                    await ws.send_json({"type": "error", "detail": _MISSING_KEY_DETAIL})
                    continue
                except ValueError as exc:
                    await ws.send_json({"type": "error", "detail": str(exc)})
                    continue
                for reply in replies:
                    await _stream_reply(ws, reply)
                await ws.send_json({"type": "scene_complete"})

            # --- Non-LLM actions: single result message. --------------------
            elif req.type in ("examine", "accuse"):
                try:
                    if req.type == "examine":
                        result = orch.examine(req.clue_id or "")
                    else:
                        result = orch.accuse_solution(
                            req.character_id or "", room=req.room, weapon=req.weapon
                        )
                except ValueError as exc:
                    await ws.send_json({"type": "error", "detail": str(exc)})
                    continue
                await ws.send_json({"type": "result", **result})

            else:
                await ws.send_json(
                    {"type": "error", "detail": f"Unknown action type '{req.type}'."}
                )
    except WebSocketDisconnect:
        return


# --------------------------------------------------------------------------- #
# Root + static assets
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def index() -> Any:
    """Serve the real frontend index if present, else a boot placeholder."""
    if _INDEX_HTML.is_file():
        return FileResponse(_INDEX_HTML)
    return HTMLResponse(_PLACEHOLDER_HTML)


# Mounted LAST so it only catches paths not handled by the routes above
# (e.g. /main.js, /styles.css). Missing files simply 404.
app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
