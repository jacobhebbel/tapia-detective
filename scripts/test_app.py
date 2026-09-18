"""Module-level smoke test for backend/app.py (no live LLM, no API key).

Uses fastapi.testclient.TestClient (an in-process ASGI client) to exercise the
REST surface and confirm the WebSocket error path. This intentionally does NOT
run a full end-to-end game with a live LLM -- only that the API layer wires to
the orchestrator correctly and degrades cleanly when OPENROUTER_API_KEY is
absent.

Run from the repo root::

    python scripts/test_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Make sure no key is present so we exercise the graceful 503/error path.
os.environ.pop("OPENROUTER_API_KEY", None)

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import app  # noqa: E402


def main() -> None:
    client = TestClient(app)

    # 1) start -> roster of 6 + evidence board.
    r = client.post("/game/start", json={"seed": 1978})
    assert r.status_code == 200, r.text
    start = r.json()
    assert len(start["roster"]) == 6, start["roster"]
    assert "evidence_board" in start and start["evidence_board"], start
    assert start["turn"] == 0, start
    print(f"[ok] POST /game/start -> {len(start['roster'])} suspects, "
          f"{len(start['evidence_board'])} clues, turn={start['turn']}")

    # 2) player-facing state keys.
    r = client.get("/game/state")
    assert r.status_code == 200, r.text
    state = r.json()
    for key in ("turn", "roster", "public_canon", "evidence_board", "transcript"):
        assert key in state, f"missing key {key!r} in /game/state"
    print(f"[ok] GET /game/state -> keys {sorted(state)}")

    # 3) GM/judge state keys.
    r = client.get("/gm/state")
    assert r.status_code == 200, r.text
    gm = r.json()
    for key in ("turn", "disclosure", "trust_graph", "evidence_board",
                "flags", "leak_violations"):
        assert key in gm, f"missing key {key!r} in /gm/state"
    assert len(gm["disclosure"]) == 6, gm["disclosure"].keys()
    print(f"[ok] GET /gm/state -> keys {sorted(gm)}")

    # 4) examine a real (initially-undiscovered) clue -> flips discovered.
    clue_id = "clue_camera_gap"
    before = {c["clue_id"]: c["discovered"] for c in start["evidence_board"]}
    assert before.get(clue_id) is False, before
    r = client.post("/game/action", json={"type": "examine", "clue_id": clue_id})
    assert r.status_code == 200, r.text
    ex = r.json()
    assert ex["clue_id"] == clue_id and ex["discovered"] is True, ex
    assert ex["newly_discovered"] is True, ex
    print(f"[ok] POST /game/action examine {clue_id} -> discovered flipped")

    # 5) correct final accusation wins; wrong killer does not.
    r = client.post(
        "/game/action",
        json={"type": "accuse", "character_id": "marrow",
              "room": "Study", "weapon": "Ammonite"},
    )
    assert r.status_code == 200, r.text
    win = r.json()
    assert win["win"] is True, win
    print(f"[ok] POST /game/action accuse(marrow, Study, Ammonite) -> win={win['win']}")

    r = client.post("/game/action", json={"type": "accuse", "character_id": "wren"})
    assert r.status_code == 200, r.text
    lose = r.json()
    assert lose["win"] is False and lose["correct_killer"] is False, lose
    print(f"[ok] POST /game/action accuse(wren) -> win={lose['win']}")

    # 6) interrogate WITHOUT an API key -> clean 503 (not a 500 stack trace).
    r = client.post(
        "/game/action",
        json={"type": "interrogate", "character_id": "marrow",
              "utterance": "Where were you?", "action_type": "ask"},
    )
    assert r.status_code == 503, f"expected 503, got {r.status_code}: {r.text}"
    body = r.json()
    assert "OPENROUTER_API_KEY" in body["detail"], body
    print(f"[ok] POST /game/action interrogate (no key) -> 503 clean error")

    # 7) 409 before a game starts is handled (fresh module state check via WS).
    with client.websocket_connect("/game/stream") as socket:
        # LLM action without a key streams an 'error' message, not a crash.
        socket.send_json({"type": "interrogate", "character_id": "marrow",
                          "utterance": "hello"})
        msg = socket.receive_json()
        assert msg["type"] == "error", msg
        assert "OPENROUTER_API_KEY" in msg["detail"], msg
        # A non-LLM action over WS returns a single result message.
        socket.send_json({"type": "examine", "clue_id": "clue_drag_marks"})
        msg = socket.receive_json()
        assert msg["type"] == "result" and msg["clue_id"] == "clue_drag_marks", msg
    print("[ok] WS /game/stream -> error on missing key, result on examine")

    print("\nAll backend/app.py smoke-test assertions PASSED.")


if __name__ == "__main__":
    main()
