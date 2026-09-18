"""Turn-based orchestrator for the Nautilus House game (plan section 5).

The :class:`Orchestrator` is the single seam the FastAPI layer will call. It
wires together the already-built modules -- it does NOT re-implement any of
their logic:

    content_loader   -> content (agent-facing) + ground truth (GM-only win check)
    CharacterAgent   -> per-suspect partitioned memory + the LLM call
    DisclosureTracker-> tiered "may this fact be revealed yet?" permission
    TrustGraph       -> per-relationship trust bands + event-driven updates
    leak_guard.scan  -> runtime safety net (imported defensively; optional)

One player action == one turn. The public method surface is deliberately
stable so the API phase can bind endpoints directly to these signatures.

Partition note: only :meth:`accuse_solution` and (indirectly) nothing else
touch the GM-only ground truth. Ground-truth strings are NEVER passed into a
CharacterAgent -- agents only ever receive agent-facing content + the
per-turn trust band + disclosure-unlocked ``allowed_facts``.

Run the module-only smoke test (no API key required) with::

    python -m backend.orchestrator
"""

from __future__ import annotations

import re
from typing import Any

# Package-relative imports (CharacterAgent itself uses `from . import ...`, so
# this module is imported/run as part of the `backend` package -- e.g. via
# `python -m backend.orchestrator`).
from . import content_loader
from .character_agent import CharacterAgent  # noqa: F401  (re-exported for typing)
from .disclosure import DisclosureTracker  # noqa: F401
from .trust_graph import TrustGraph  # noqa: F401
from .game_state import GameState

# --- Leak-guard: imported defensively so build order doesn't matter. --------
# leak_guard.py may be built in parallel; if it's missing we fall back to a
# no-op scanner so the orchestrator still runs end-to-end.
try:
    from .leak_guard import scan as _leak_scan  # type: ignore
except Exception:  # pragma: no cover - fallback when leak_guard isn't built yet
    try:
        from leak_guard import scan as _leak_scan  # type: ignore
    except Exception:
        def _leak_scan(character_id, text, *, disclosure_state=None):  # type: ignore
            return []


# ---------------------------------------------------------------------------
# Tunables / constants
# ---------------------------------------------------------------------------

ACTION_TYPES = ("ask", "present_evidence", "accuse")

# The single supporting clue whose explanation *clears* Wren. Discovering it is
# the in-game trigger that flips the "wren_innocent" flag, which is the ONLY
# thing that collapses Hettie's protect-Wren goal and unlocks her T3 secret.
WREN_EXONERATION_CLUE = "clue_wren_arrival_discrepancy"
WREN_INNOCENT_FLAG = "wren_innocent"

# Cap on auto agent-to-agent replies in a multi-character scene (plan: ~2-3).
SCENE_AUTO_REPLY_CAP = 3

# Phrases that read as the player offering a trade/protection/silence => trade
# pressure for the disclosure ladder.
_TRADE_PHRASES: tuple[str, ...] = (
    "protect you",
    "i'll protect",
    "keep it secret",
    "keep this secret",
    "keep this between us",
    "keep it between us",
    "won't tell",
    "will not tell",
    "i promise",
    "you have my word",
    "in exchange",
    "make a deal",
    "cut you a deal",
    "immunity",
    "stay silent",
    "keep quiet",
    "cover for you",
    "off the record",
    "no one will know",
    "nobody will know",
    "i can help you",
)

# Minimal stopword set for the clue-detection heuristic (kept transparent).
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "that", "this", "with", "was", "were", "she",
        "her", "his", "him", "them", "they", "you", "your", "not", "but",
        "had", "has", "have", "did", "does", "are", "who", "what", "when",
        "where", "why", "how", "which", "into", "from", "out", "about",
        "some", "any", "all", "one", "two", "been", "being", "its", "our",
        "their", "there", "here", "then", "than", "just", "can", "him",
        "night", "room", "vance", "body", "were", "found", "around", "that",
        "about", "would", "could", "should", "been",
    }
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Minimum number of distinctive clue keywords that must appear in a line for it
# to count as "naming" that clue.
_CLUE_MATCH_THRESHOLD = 2


def _keywords(text: str) -> set[str]:
    """Distinctive lowercased tokens (len >= 4, non-stopword) for matching."""
    if not text:
        return set()
    return {
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) >= 4 and tok not in _STOPWORDS
    }


class Orchestrator:
    """Drives the turn loop over the shared :class:`GameState`."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.state = GameState.new(seed)

        # Precompute distinctive keyword sets per clue for detection. Anchor +
        # supporting clues both live in the evidence board.
        self._clue_keywords: dict[str, set[str]] = {}
        for clue_id, clue in self.state.evidence_board.items():
            text = " ".join(
                [clue.get("statement", "")] + list(clue.get("proves", []))
            )
            self._clue_keywords[clue_id] = _keywords(text)

    # ================================================================== #
    # Public API (stable surface for the FastAPI phase)
    # ================================================================== #

    def start(self) -> dict:
        """Return the initial player-facing state.

        Shape::

            {
              "roster": [{"id", "name", "surface"}, ...],   # 6 suspects
              "public_canon": {...},                         # setting + anchors
              "evidence_board": [{"clue_id","category",
                                  "discovered","statement"}, ...],
              "turn": 0,
            }
        """
        return {
            "roster": self.state.roster(),
            "public_canon": content_loader.load_public_canon(),
            "evidence_board": self.state.public_evidence(),
            "turn": self.state.turn,
        }

    def interrogate(
        self,
        character_id: str,
        utterance: str,
        action_type: str = "ask",
    ) -> dict:
        """Run a single-character turn (plan section 5 pipeline).

        Returns::

            {
              "speaker": <character_id>,
              "text": <the character's reply>,
              "turn": <turn counter AFTER this action>,
              "newly_revealed": [{"fact_id", "summary"}, ...],
              "trust_band": <band of this character toward the player>,
              "violations": [<leak-guard hit>, ...],
            }
        """
        self._validate_character(character_id)
        if action_type not in ACTION_TYPES:
            raise ValueError(
                f"Unknown action_type '{action_type}'. Expected one of: "
                f"{', '.join(ACTION_TYPES)}"
            )

        # A single interrogation is a one-character scene.
        self.state.present_characters = [character_id]

        result = self._elicit(
            character_id=character_id,
            speaker="player",
            utterance=utterance,
            action_type=action_type,
            present=[character_id],
            turn=self.state.turn,
        )

        # Step 8: fire a trust event if this action was an accusation.
        if action_type == "accuse":
            if self._detect_clues(utterance):
                # Evidence-backed accusation lands: broad trust erosion.
                self.state.trust_graph.apply_event(
                    {
                        "type": "accusation_backed",
                        "actor": "player",
                        "target": character_id,
                        "present": ["player", character_id],
                    }
                )
            else:
                # Unsupported accusation: mutual distrust between the two.
                self.state.trust_graph.apply_event(
                    {
                        "type": "accusation",
                        "actor": "player",
                        "target": character_id,
                    }
                )

        # Step 9: advance the turn counter and log the transcript.
        self.state.turn += 1
        result["turn"] = self.state.turn
        self.state.transcript.append(
            {
                "turn": self.state.turn,
                "type": "interrogate",
                "action_type": action_type,
                "player_utterance": utterance,
                "speaker": character_id,
                "text": result["text"],
                "newly_revealed": result["newly_revealed"],
                "violations": result["violations"],
            }
        )
        return result

    def scene(self, character_ids: list[str], utterance: str) -> list[dict]:
        """Run a multi-character scene (plan section 5).

        The player's line goes to every present agent (phase 1); then a BOUNDED
        agent-to-agent chain (<= ``SCENE_AUTO_REPLY_CAP`` auto-replies) lets
        present agents react to what another just said. Revealed facts are
        broadcast between present agents via ``observe(...)`` as they surface.

        Returns the ordered list of per-utterance dicts, each shaped like an
        :meth:`interrogate` result (``speaker``/``text``/``turn``/
        ``newly_revealed``/``trust_band``/``violations``).
        """
        if not character_ids:
            raise ValueError("scene() requires at least one character_id.")
        for cid in character_ids:
            self._validate_character(cid)

        present = list(character_ids)
        self.state.present_characters = present
        turn = self.state.turn
        results: list[dict] = []

        # --- Phase 1: the player's line goes to every present agent. --------
        for cid in present:
            res = self._elicit(
                character_id=cid,
                speaker="player",
                utterance=utterance,
                action_type="ask",
                present=present,
                turn=turn,
            )
            results.append(res)

        # --- Phase 2: bounded agent-to-agent reaction chain. ----------------
        # Each auto-reply: pick the next present agent (round-robin, skipping the
        # last speaker) and let them react to the most recent utterance.
        if len(present) >= 2 and results:
            last_speaker = results[-1]["speaker"]
            last_text = results[-1]["text"]
            order_pos = present.index(last_speaker)
            for _ in range(SCENE_AUTO_REPLY_CAP):
                # Advance round-robin to the next agent that isn't the speaker.
                order_pos = (order_pos + 1) % len(present)
                responder = present[order_pos]
                if responder == last_speaker:
                    continue
                res = self._elicit(
                    character_id=responder,
                    speaker=last_speaker,
                    utterance=last_text,
                    action_type="ask",
                    present=present,
                    turn=turn,
                )
                results.append(res)
                last_speaker = responder
                last_text = res["text"]

        # One player action == one turn.
        self.state.turn += 1
        for res in results:
            res["turn"] = self.state.turn
        self.state.transcript.append(
            {
                "turn": self.state.turn,
                "type": "scene",
                "present": present,
                "player_utterance": utterance,
                "utterances": [
                    {
                        "speaker": r["speaker"],
                        "text": r["text"],
                        "newly_revealed": r["newly_revealed"],
                        "violations": r["violations"],
                    }
                    for r in results
                ],
            }
        )
        return results

    def examine(self, clue_id: str) -> dict:
        """Mark a clue discovered on the evidence board (no LLM call).

        If the discovered clue is the one that exonerates Wren, set the
        ``wren_innocent`` flag -- the trigger that later unlocks Hettie's T3.
        """
        if clue_id not in self.state.evidence_board:
            raise ValueError(f"Unknown clue_id '{clue_id}'.")

        clue = self.state.evidence_board[clue_id]
        already = clue.get("discovered", False)
        clue["discovered"] = True

        wren_flag_set = False
        if clue_id == WREN_EXONERATION_CLUE:
            self.state.flags.add(WREN_INNOCENT_FLAG)
            wren_flag_set = True

        self.state.turn += 1
        self.state.transcript.append(
            {
                "turn": self.state.turn,
                "type": "examine",
                "clue_id": clue_id,
                "newly_discovered": not already,
            }
        )
        return {
            "clue_id": clue_id,
            "discovered": True,
            "newly_discovered": not already,
            "statement": clue.get("statement"),
            "wren_innocent_flag_set": wren_flag_set,
            "flags": sorted(self.state.flags),
            "turn": self.state.turn,
        }

    def set_flag(self, flag: str, value: bool = True) -> None:
        """Explicitly set/clear a game flag (e.g. ``"wren_innocent"``).

        Provided so a caller can flip the Wren-innocent flag without going
        through :meth:`examine` (the flag is what inverts Hettie's goal stack).
        """
        if value:
            self.state.flags.add(flag)
        else:
            self.state.flags.discard(flag)

    def accuse_solution(
        self,
        character_id: str,
        room: str | None = None,
        weapon: str | None = None,
    ) -> dict:
        """Resolve a final accusation against the GM-only ground truth.

        Compares the accused killer (and optionally the room/weapon) to
        ``gm_load_ground_truth()``. A win requires the correct killer, plus any
        supplied room/weapon also being correct. Returns which parts matched.
        """
        gt = content_loader.gm_load_ground_truth()
        true_killer = gt["killer"]["character_id"]
        true_room = gt["true_murder_room"]
        true_weapon = gt["weapon"]["name"]

        correct_killer = character_id == true_killer
        correct_room = None if room is None else _norm(room) == _norm(true_room)
        correct_weapon = None if weapon is None else _weapon_match(weapon, true_weapon)

        win = (
            correct_killer
            and (correct_room is not False)
            and (correct_weapon is not False)
        )

        self.state.turn += 1
        self.state.transcript.append(
            {
                "turn": self.state.turn,
                "type": "accuse_solution",
                "accused": character_id,
                "room": room,
                "weapon": weapon,
                "win": win,
            }
        )
        return {
            "accused": character_id,
            "correct_killer": correct_killer,
            "correct_room": correct_room,
            "correct_weapon": correct_weapon,
            "win": win,
            "turn": self.state.turn,
            "solution": {
                "killer": true_killer,
                "room": true_room,
                "weapon": true_weapon,
            },
        }

    def gm_state(self) -> dict:
        """Return the judge/GM view: disclosure state for all characters, the
        trust map, the full evidence board, and accumulated leak violations.
        """
        return self.state.to_gm_dict()

    def public_state(self) -> dict:
        """Convenience: the full player-facing snapshot (delegates to state)."""
        return self.state.to_public_dict()

    # ================================================================== #
    # Internal: the per-utterance pressure pipeline
    # ================================================================== #

    def _elicit(
        self,
        *,
        character_id: str,
        speaker: str,
        utterance: str,
        action_type: str,
        present: list[str],
        turn: int,
    ) -> dict:
        """Run the full per-utterance pipeline for one character and return the
        per-utterance result dict. Does NOT advance ``self.state.turn`` (the
        caller owns turn accounting so a scene can be one player turn).
        """
        agent = self.state.agents[character_id]
        tracker = self.state.trackers[character_id]

        # 2) Trust band: this character's stance toward whoever is speaking.
        trust_band = self.state.trust_graph.band(character_id, speaker)

        # 1) Detect named clues + treat the whole line as the asked topic.
        clues_named = self._detect_clues(utterance)
        asked_topics = [utterance] if utterance else []
        trade_offered = self._detect_trade(utterance)

        # accuse => social pressure from the speaker; only the player can
        # "accuse" via interrogate, but agent-to-agent accusations flow here too.
        present_accusers = [speaker] if action_type == "accuse" else None

        # Hettie's T3 only inverts once Wren is proven innocent.
        goal_stack_inverted = (
            character_id == "hettie" and WREN_INNOCENT_FLAG in self.state.flags
        )

        # 3) Register pressure with the disclosure ladder.
        allowed_before = set(tracker.allowed_facts())
        tracker.register_pressure(
            utterance=utterance,
            asked_topics=asked_topics,
            present_accusers=present_accusers,
            clues_named=clues_named or None,
            trade_offered=trade_offered,
            trust_band=trust_band,
            goal_stack_inverted=goal_stack_inverted,
        )

        # 4) Currently-unlocked facts (post-pressure).
        allowed = tracker.allowed_facts()
        newly_ids = [fid for fid in allowed if fid not in allowed_before]

        # Keep the agent's introspective disclosure_state in sync with the
        # authoritative tracker so the leak-guard can tell what's legitimately
        # disclosable now.
        self._sync_disclosure_state(character_id)

        # 5) The LLM call (may raise MissingAPIKeyError upstream if no key).
        text = agent.respond(
            speaker,
            utterance,
            trust_band=trust_band,
            allowed_facts=allowed,
            present_characters=present,
        )
        if not isinstance(text, str):  # defensive: coalesce a stream iterator
            text = "".join(text)

        # 6) Leak-guard safety net; record any violations for the GM panel.
        violations = _leak_scan(
            character_id, text, disclosure_state=agent.disclosure_state
        )
        if violations:
            self.state.leak_violations.append(
                {
                    "turn": turn,
                    "character_id": character_id,
                    "text": text,
                    "violations": violations,
                }
            )

        # 7) Record the character's own turn; broadcast newly-stated facts to
        #    the other present characters' heard_from_others.
        agent.remember(
            {
                "turn": turn,
                "asked_by": speaker,
                "asked": utterance,
                "statement": text,
            }
        )
        newly_revealed = [
            {"fact_id": fid, "summary": self._secret_summary(character_id, fid)}
            for fid in newly_ids
        ]
        self._broadcast(source_id=character_id, present=present, text=text,
                        revealed=newly_revealed, turn=turn)

        return {
            "speaker": character_id,
            "text": text,
            "turn": turn,
            "newly_revealed": newly_revealed,
            "trust_band": trust_band,
            "violations": violations,
        }

    # ================================================================== #
    # Internal helpers
    # ================================================================== #

    def _validate_character(self, character_id: str) -> None:
        if character_id not in self.state.agents:
            raise ValueError(
                f"Unknown character_id '{character_id}'. Expected one of: "
                f"{', '.join(self.state.agents)}"
            )

    def _detect_clues(self, text: str) -> list[str]:
        """Substring/keyword match a line against the clue board.

        A clue is considered "named" if its clue_id appears literally, or if at
        least ``_CLUE_MATCH_THRESHOLD`` of its distinctive keywords appear in
        the line. Returns the matching clue_ids.
        """
        if not text:
            return []
        low = text.lower()
        line_kw = _keywords(text)
        named: list[str] = []
        for clue_id, kw in self._clue_keywords.items():
            if clue_id.lower() in low:
                named.append(clue_id)
                continue
            if kw and len(line_kw & kw) >= _CLUE_MATCH_THRESHOLD:
                named.append(clue_id)
        return named

    @staticmethod
    def _detect_trade(text: str) -> bool:
        low = (text or "").lower()
        return any(phrase in low for phrase in _TRADE_PHRASES)

    def _secret_summary(self, character_id: str, fact_id: str) -> str:
        """Look up a secret's summary from the character's own dossier data."""
        for secret in self.state.agents[character_id].secrets:
            if secret.get("fact_id") == fact_id:
                return secret.get("summary", "")
        return ""

    def _sync_disclosure_state(self, character_id: str) -> None:
        """Mirror the DisclosureTracker's per-tier state onto the agent's
        introspective ``disclosure_state`` (fact_id -> "sealed"/"pressured_once"
        /"revealed"), so leak_guard has an accurate view of what's disclosable.
        """
        agent = self.state.agents[character_id]
        tier_state = self.state.trackers[character_id].state().get("tiers", {})
        for tier in tier_state.values():
            fid = tier.get("fact_id")
            if fid is not None:
                agent.disclosure_state[fid] = tier.get("state", "sealed")

    def _broadcast(
        self,
        *,
        source_id: str,
        present: list[str],
        text: str,
        revealed: list[dict],
        turn: int,
    ) -> None:
        """Broadcast newly-stated facts to the OTHER present characters.

        Facts broadcast = the secret summaries just unlocked by ``source_id`` +
        any clue-board facts detected in the spoken line. Each is appended to
        the listeners' ``heard_from_others`` via ``observe(...)``.
        """
        others = [cid for cid in present if cid != source_id]
        if not others:
            return

        source_name = self.state.agents[source_id].name
        statements: list[str] = [r["summary"] for r in revealed if r.get("summary")]
        for clue_id in self._detect_clues(text):
            stmt = self.state.evidence_board.get(clue_id, {}).get("statement")
            if stmt:
                statements.append(stmt)

        if not statements:
            return
        for listener_id in others:
            listener = self.state.agents[listener_id]
            for stmt in statements:
                listener.observe(
                    {"source": source_name, "turn": turn, "statement": stmt}
                )


def _norm(value: str) -> str:
    """Lowercase, strip, and drop a leading article for lenient matching."""
    v = (value or "").strip().lower()
    for article in ("the ", "a ", "an "):
        if v.startswith(article):
            v = v[len(article):]
    return v


def _weapon_match(guess: str, truth: str) -> bool:
    """Lenient weapon match: article-insensitive substring either direction
    (so "Ammonite" matches ground truth "the Ammonite")."""
    g, t = _norm(guess), _norm(truth)
    return bool(g) and (g in t or t in g)


# ==========================================================================
# Module-only smoke test (NO API key required). Run:
#     python -m backend.orchestrator
# It monkeypatches the LLM client with a deterministic fake so the whole
# pipeline (disclosure + trust + broadcast + leak-guard + win check) runs
# offline. This is NOT the end-to-end demo (that's a later phase).
# ==========================================================================
if __name__ == "__main__":
    from . import llm_client

    # --- Deterministic fake LLM: echoes a canned, leak-safe line. -----------
    _CALLS: list[dict] = []

    def _fake_chat(messages, model=None, temperature=0.8, stream=False):
        _CALLS.append({"messages": messages})
        # A bland, in-character-ish line that names no forbidden content.
        return "I've told you what I can. Ask me plainly and I'll answer."

    llm_client.chat = _fake_chat  # CharacterAgent.respond calls llm_client.chat

    print("=" * 74)
    print("ORCHESTRATOR SMOKE TEST (offline, faked LLM)")
    print("=" * 74)

    orch = Orchestrator(seed=1978)

    start = orch.start()
    assert len(start["roster"]) == 6, "roster should list six suspects"
    assert start["turn"] == 0
    print(f"start(): {len(start['roster'])} suspects, "
          f"{len(start['evidence_board'])} clue slots, turn={start['turn']}")

    # --- A few interrogate turns on Marrow (the highest-risk leak case). ----
    r1 = orch.interrogate(
        "marrow",
        "Doctor, when did you last see Vance alone that night?",
        "ask",
    )
    assert r1["turn"] == 1, r1["turn"]
    print(f"\nturn {r1['turn']} ask -> band={r1['trust_band']}, "
          f"newly_revealed={[f['fact_id'] for f in r1['newly_revealed']]}")

    r2 = orch.interrogate(
        "marrow",
        "You saw Vance late, giving him his evening medication -- tell me "
        "about that last visit when you were alone with him.",
        "ask",
    )
    assert r2["turn"] == 2, r2["turn"]
    print(f"turn {r2['turn']} ask -> band={r2['trust_band']}, "
          f"newly_revealed={[f['fact_id'] for f in r2['newly_revealed']]}")

    # Disclosure state must have advanced beyond fully-sealed for Marrow.
    marrow_state = orch.state.trackers["marrow"].state()
    marrow_tiers = {k: v["state"] for k, v in marrow_state["tiers"].items()}
    print(f"marrow disclosure tiers: {marrow_tiers}")
    assert "marrow_secret_t1_last_seen" in orch.state.trackers["marrow"].allowed_facts(), \
        "Marrow's T1 (last-seen) should unlock after direct questioning"

    # --- An accusation should move trust (Marrow -> player drops to wary). --
    band_before = orch.state.trust_graph.band("marrow", "player")
    r3 = orch.interrogate("marrow", "I think you killed Vance.", "accuse")
    band_after = orch.state.trust_graph.band("marrow", "player")
    assert r3["turn"] == 3, r3["turn"]
    print(f"\nturn {r3['turn']} accuse -> marrow->player trust band "
          f"{band_before} -> {band_after} "
          f"(score={orch.state.trust_graph.score('marrow','player'):+.2f})")
    assert band_after == "wary", f"expected 'wary' after accusation, got {band_after}"

    # --- Examine the Wren-exoneration clue -> flag flips. -------------------
    ex = orch.examine(WREN_EXONERATION_CLUE)
    assert ex["wren_innocent_flag_set"] is True
    assert WREN_INNOCENT_FLAG in orch.state.flags
    print(f"\nexamine({WREN_EXONERATION_CLUE}) -> wren_innocent flag set: "
          f"{WREN_INNOCENT_FLAG in orch.state.flags}")

    # --- A quick multi-character scene (bounded agent-to-agent chain). ------
    scene = orch.scene(["hettie", "slate"], "Were the two of you together at all that night?")
    speakers = [u["speaker"] for u in scene]
    print(f"\nscene(hettie, slate) produced {len(scene)} utterances: {speakers}")
    assert len(scene) >= 2, "scene should include at least both present agents"

    # --- Win-check: correct solution wins; wrong killer does not. -----------
    win = orch.accuse_solution("marrow", room="Study", weapon="Ammonite")
    print(f"\naccuse_solution(marrow, Study, Ammonite) -> "
          f"win={win['win']} (killer={win['correct_killer']}, "
          f"room={win['correct_room']}, weapon={win['correct_weapon']})")
    assert win["win"] is True, win

    lose = orch.accuse_solution("wren")
    print(f"accuse_solution(wren) -> win={lose['win']} "
          f"(correct_killer={lose['correct_killer']})")
    assert lose["win"] is False and lose["correct_killer"] is False, lose

    # --- GM panel dump. -----------------------------------------------------
    gm = orch.gm_state()
    print("\n" + "=" * 74)
    print("gm_state() snapshot")
    print("=" * 74)
    print(f"turn: {gm['turn']}")
    print("disclosure (per character, tier states):")
    for cid, st in gm["disclosure"].items():
        tiers = {k: v["state"] for k, v in st["tiers"].items()}
        print(f"  {cid:8s}: {tiers}")
    print(f"trust_graph nodes: {len(gm['trust_graph'])}")
    print(f"  marrow->player: {gm['trust_graph']['marrow']['player']:+.2f}")
    print(f"flags: {gm['flags']}")
    print(f"evidence discovered: "
          f"{[cid for cid, c in gm['evidence_board'].items() if c['discovered']]}")
    print(f"leak_violations: {len(gm['leak_violations'])}")

    print("\nAll orchestrator smoke-test assertions PASSED.")
