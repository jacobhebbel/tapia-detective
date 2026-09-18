"""Shared game state for the Nautilus House game.

This module holds the single-session, in-memory state the orchestrator mutates
each turn (per plan section 5). It deliberately owns *data + serialization*
only -- the turn loop / pressure pipeline lives in `orchestrator.py`.

What a :class:`GameState` bundles:
  * ``turn``              -- monotonic player-action counter.
  * ``agents``            -- ``{character_id: CharacterAgent}`` (partitioned memory).
  * ``trackers``          -- ``{character_id: DisclosureTracker}`` (tiered unlocks).
  * ``trust_graph``       -- one shared :class:`TrustGraph` over 6 chars + player.
  * ``evidence_board``    -- ``{clue_id: {..clue.., "discovered": bool}}`` seeded
                             from ``content_loader.load_clues()``.
  * ``transcript``        -- ordered list of turn records for the UI.
  * ``present_characters``-- ids in the current scene.
  * ``flags``             -- game flags (e.g. ``"wren_innocent"``).
  * ``leak_violations``   -- accumulated leak-guard hits for the GM panel.

Serialization:
  * :meth:`to_public_dict` -- safe for the player-facing UI (no ground truth,
    no disclosure internals, only *discovered* clue statements).
  * :meth:`to_gm_dict`     -- the judge/GM view: full disclosure ``state()``,
    the raw trust map, the whole evidence board, and any leak violations.

Standard library + sibling backend modules only -- no new pip dependencies.
Run the smoke test in ``orchestrator.py`` (``python -m backend.orchestrator``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Support both package import (`from backend import game_state`) and running
# a sibling module directly. CharacterAgent uses package-relative imports, so
# the smoke test is expected to run via `python -m backend.orchestrator`.
try:  # package import
    from . import content_loader
    from .character_agent import CharacterAgent
    from .disclosure import DisclosureTracker
    from .trust_graph import TrustGraph
except ImportError:  # pragma: no cover - script fallback
    import content_loader  # type: ignore[no-redef]
    from character_agent import CharacterAgent  # type: ignore[no-redef]
    from disclosure import DisclosureTracker  # type: ignore[no-redef]
    from trust_graph import TrustGraph  # type: ignore[no-redef]


@dataclass
class GameState:
    """In-memory state for a single live session."""

    turn: int
    agents: dict[str, CharacterAgent]
    trackers: dict[str, DisclosureTracker]
    trust_graph: TrustGraph
    evidence_board: dict[str, dict[str, Any]]
    transcript: list[dict] = field(default_factory=list)
    present_characters: list[str] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)
    leak_violations: list[dict] = field(default_factory=list)
    seed: int | None = None

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def new(cls, seed: int | None = None) -> "GameState":
        """Instantiate all six agents, trackers, the trust graph, and the
        evidence board seeded from ``content_loader.load_clues()``.

        The evidence board carries every anchor + supporting clue, each with a
        ``discovered`` flag; anchor clues marked ``discovered_by_default`` in
        the content start discovered.
        """
        character_ids = content_loader.load_all_character_ids()

        agents = {cid: CharacterAgent(cid) for cid in character_ids}
        trackers = {cid: DisclosureTracker(cid) for cid in character_ids}
        trust_graph = TrustGraph()

        evidence_board: dict[str, dict[str, Any]] = {}
        clues = content_loader.load_clues()
        for group in ("anchor_clues", "supporting_clues"):
            for clue in clues.get(group, []):
                clue_id = clue.get("clue_id")
                if not clue_id:
                    continue
                entry = dict(clue)
                entry["discovered"] = bool(clue.get("discovered_by_default", False))
                evidence_board[clue_id] = entry

        return cls(
            turn=0,
            agents=agents,
            trackers=trackers,
            trust_graph=trust_graph,
            evidence_board=evidence_board,
            transcript=[],
            present_characters=[],
            flags=set(),
            leak_violations=[],
            seed=seed,
        )

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def roster(self) -> list[dict]:
        """Player-facing roster: id, display name, and public surface blurb."""
        return [
            {
                "id": cid,
                "name": agent.name,
                "surface": agent.static.get("surface", ""),
            }
            for cid, agent in self.agents.items()
        ]

    def public_evidence(self) -> list[dict]:
        """Evidence board for the player UI.

        Undiscovered clues appear as locked slots (no statement) so the UI can
        show progress without spoiling their contents.
        """
        board = []
        for clue_id, clue in self.evidence_board.items():
            discovered = clue.get("discovered", False)
            board.append(
                {
                    "clue_id": clue_id,
                    "category": clue.get("category"),
                    "discovered": discovered,
                    "statement": clue.get("statement") if discovered else None,
                }
            )
        return board

    def to_public_dict(self) -> dict:
        """Player-facing snapshot -- no ground truth, no disclosure internals."""
        return {
            "turn": self.turn,
            "present_characters": list(self.present_characters),
            "roster": self.roster(),
            "public_canon": content_loader.load_public_canon(),
            "evidence_board": self.public_evidence(),
            "transcript": list(self.transcript),
        }

    def to_gm_dict(self) -> dict:
        """Judge/GM snapshot -- disclosure state, raw trust map, leak log.

        This is the only serialization that exposes per-character disclosure
        internals and the full trust graph. It never inlines ground_truth.json
        (that stays in the accusation-resolution path), but it does surface
        everything a judge needs to watch secrets held / leaked in real time.
        """
        return {
            "turn": self.turn,
            "present_characters": list(self.present_characters),
            "disclosure": {
                cid: tracker.state() for cid, tracker in self.trackers.items()
            },
            "trust_graph": self.trust_graph.to_dict(),
            "evidence_board": {
                clue_id: {
                    "category": clue.get("category"),
                    "discovered": clue.get("discovered", False),
                    "statement": clue.get("statement"),
                }
                for clue_id, clue in self.evidence_board.items()
            },
            "flags": sorted(self.flags),
            "leak_violations": list(self.leak_violations),
        }
