"""Trust graph for the Nautilus House game.

===============================================================================
WHAT THIS IS
===============================================================================
A tiny, self-contained directed weighted trust graph. Nodes are the six
characters plus ``"player"``. An edge ``trust[A][B]`` in ``[-1.0, 1.0]`` is
A's trust *toward* B (directed -- A->B and B->A can differ).

The orchestrator fires ``apply_event(...)`` after each turn to nudge trust,
and pulls ``band(...)`` / ``directive(...)`` to inject a single natural-language
stance line into a character's prompt. Raw floats are never shown to the LLM --
personas roleplay discrete bands far better than numbers.

Design constraints (per plan section 4):
  * Standard library ONLY -- no pip installs, no LLM calls.
  * Does NOT import character_agent.py or disclosure.py (built in parallel);
    this module is intentionally free of any game-engine coupling.

Seed values live in the ``DEFAULT_SEEDS`` constant below so they are easy to
tune in one place.
===============================================================================
"""

from __future__ import annotations

from typing import Iterable

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

# The six characters (must match content_loader.CHARACTER_IDS) + the player.
CHARACTER_IDS: tuple[str, ...] = (
    "odette",
    "barnaby",
    "slate",
    "marrow",
    "wren",
    "hettie",
)
PLAYER_ID = "player"
NODES: tuple[str, ...] = CHARACTER_IDS + (PLAYER_ID,)

# Human-readable names for the natural-language stance lines. Kept local so the
# module has no dependency on content_loader / the dossier files.
DISPLAY_NAMES: dict[str, str] = {
    "odette": "Odette",
    "barnaby": "Barnaby",
    "slate": "Colonel Slate",
    "marrow": "Dr. Marrow",
    "wren": "Wren",
    "hettie": "Hettie",
    "player": "the detective",
}

# ---------------------------------------------------------------------------
# Seed values -- TUNE HERE
# ---------------------------------------------------------------------------
# Each entry is (from_id, to_id, trust_value). Any pair not listed defaults to
# 0.0 (neutral), and the player starts neutral (0.0) toward/from everyone.
#
# Grounding (all from the cast dossiers / plan section 4):
#   * Hettie & Slate are secretly in a relationship -> strong mutual trust.
#   * Hettie is protecting Wren (thinks Wren is the killer) -> high Hettie->Wren.
#   * Wren wrongly suspects Barnaby; Barnaby wrongly suspects Odette;
#     Odette gleefully floats "Wren did it" -> mild wariness among them.
#   * Slate has a vague hunch Marrow is "hiding something" -> mild wariness.
#   * Marrow keeps up the affable-caretaker act toward everyone -> faint warmth.
DEFAULT_SEEDS: tuple[tuple[str, str, float], ...] = (
    # --- Hettie <-> Slate: the protected, mutual relationship ---
    ("hettie", "slate", 0.9),
    ("slate", "hettie", 0.9),
    # --- Hettie -> Wren: she is actively protecting Wren ---
    ("hettie", "wren", 0.8),
    # Wren doesn't know Hettie is shielding her, but Hettie is a warm,
    # long-trusted house fixture -> mild positive back.
    ("wren", "hettie", 0.2),
    # --- Rivals / mutual suspicion (mild negatives) ---
    ("wren", "barnaby", -0.3),   # Wren's ungrounded hunch about Barnaby
    ("barnaby", "odette", -0.3),  # Barnaby thinks Odette could do something drastic
    ("odette", "wren", -0.2),     # Odette performatively pins it on Wren
    ("slate", "marrow", -0.2),    # Slate's vague "he's hiding something" hunch
    # --- Marrow's helpful-caretaker veneer: faint warmth outward ---
    ("marrow", "odette", 0.1),
    ("marrow", "wren", 0.1),
)

# ---------------------------------------------------------------------------
# Band thresholds & event deltas -- TUNE HERE
# ---------------------------------------------------------------------------

# Discretization cutoffs, checked in order hostile -> close (see band()).
_BAND_UPPER: dict[str, float] = {
    "hostile": -0.5,   # score <= -0.5
    "wary": -0.15,     # -0.5 < score <= -0.15
    "neutral": 0.15,   # -0.15 < score < 0.15
    "warm": 0.5,       # 0.15 <= score < 0.5
    # "close": score >= 0.5
}

# One stance line per band, ``{name}`` filled with the target's display name.
_BAND_DIRECTIVES: dict[str, str] = {
    "hostile": (
        "Your current stance toward {name}: HOSTILE -- you actively distrust "
        "them, treat their questions as attacks, and give up nothing willingly."
    ),
    "wary": (
        "Your current stance toward {name}: WARY -- cooperative on the surface "
        "but you don't volunteer anything that could hurt you or them without "
        "being asked twice."
    ),
    "neutral": (
        "Your current stance toward {name}: NEUTRAL -- polite and businesslike; "
        "you answer what's asked but form no special loyalty either way."
    ),
    "warm": (
        "Your current stance toward {name}: WARM -- you're inclined to trust "
        "them and will share more readily, though not your deepest secrets."
    ),
    "close": (
        "Your current stance toward {name}: CLOSE -- you trust and want to "
        "protect them; you'll cover for them and shade the truth to keep them "
        "safe."
    ),
}

# Magnitude of trust movement per event type.
DELTA_ACCUSATION = 0.25       # unsupported accusation
DELTA_ACCUSATION_BACKED = 0.15  # evidence-backed accusation, per bystander
DELTA_PROMISE = 0.3           # player keeps/breaks an in-scene promise
DELTA_DEFENSE = 0.2           # seen defending/covering for another


def _clamp(value: float) -> float:
    """Clamp a trust value into the valid [-1.0, 1.0] range."""
    return max(-1.0, min(1.0, value))


class TrustGraph:
    """Directed, weighted trust graph over the six characters + the player.

    ``trust[A][B]`` is A's trust toward B, a float in ``[-1.0, 1.0]``.
    """

    def __init__(self, seeds: dict | None = None) -> None:
        """Build the graph.

        All edges start neutral (0.0). If ``seeds`` is None, the module-level
        ``DEFAULT_SEEDS`` are applied. If ``seeds`` is provided it must be a
        nested mapping ``{from_id: {to_id: value}}`` and fully replaces the
        default seeding (any omitted pair stays neutral). The player always
        starts neutral toward and from everyone unless explicitly seeded.
        """
        # Initialize every ordered pair to 0.0 (no self-edges).
        self._trust: dict[str, dict[str, float]] = {
            a: {b: 0.0 for b in NODES if b != a} for a in NODES
        }

        if seeds is None:
            for a, b, value in DEFAULT_SEEDS:
                self._set(a, b, value)
        else:
            for a, targets in seeds.items():
                for b, value in targets.items():
                    self._set(a, b, float(value))

    # -- internal helpers ---------------------------------------------------

    def _validate(self, node: str) -> None:
        if node not in self._trust:
            raise ValueError(
                f"Unknown node '{node}'. Expected one of: {', '.join(NODES)}"
            )

    def _set(self, a: str, b: str, value: float) -> None:
        self._validate(a)
        self._validate(b)
        if a == b:
            raise ValueError(f"Self-trust edge is not allowed ('{a}' -> '{b}').")
        self._trust[a][b] = _clamp(value)

    def _adjust(self, a: str, b: str, delta: float) -> None:
        """Nudge trust[a][b] by delta, clamped. No-op for a == b."""
        if a == b:
            return
        self._validate(a)
        self._validate(b)
        self._trust[a][b] = _clamp(self._trust[a][b] + delta)

    # -- queries ------------------------------------------------------------

    def score(self, a: str, b: str) -> float:
        """Return A's trust toward B, a float in [-1.0, 1.0]."""
        self._validate(a)
        self._validate(b)
        if a == b:
            return 1.0  # trivially trusts self; not a stored edge
        return self._trust[a][b]

    def band(self, a: str, b: str) -> str:
        """Discretize score(a, b) into one of the five trust bands.

        Returns one of: hostile / wary / neutral / warm / close.
        """
        s = self.score(a, b)
        if s <= _BAND_UPPER["hostile"]:   # <= -0.5
            return "hostile"
        if s <= _BAND_UPPER["wary"]:      # <= -0.15
            return "wary"
        if s < _BAND_UPPER["neutral"]:    # < 0.15
            return "neutral"
        if s < _BAND_UPPER["warm"]:       # < 0.5
            return "warm"
        return "close"

    def directive(self, a: str, b: str) -> str:
        """Return the single natural-language stance line for A toward B.

        This is the one line the orchestrator injects into A's persona prompt.
        """
        name = DISPLAY_NAMES.get(b, b)
        return _BAND_DIRECTIVES[self.band(a, b)].format(name=name)

    # -- mutation -----------------------------------------------------------

    def apply_event(self, event: dict) -> None:
        """Mutate trust per the plan's update rules.

        Event schema (extra keys are ignored)::

            {
              "type": "accusation" | "accusation_backed"
                      | "promise_kept" | "promise_broken" | "defense",
              "actor":  <node id>,        # who did the thing
              "target": <node id>,        # who it was directed at
              "present": [<node id>, ...] # everyone in the scene (optional)
            }

        Rules:
          * accusation (unsupported): the accuser distrusts the accused more,
            and the accused resents the accuser -> both directions drop.
          * accusation_backed (lands with evidence): the accused loses trust
            broadly -- every other person present trusts them a little less.
          * promise_kept / promise_broken: trust *toward the player* (the actor)
            moves for the recipient(s) -- the target if given, otherwise
            everyone else present.
          * defense (seen covering for another): actor <-> target mutual trust
            nudges up.

        All values are clamped to [-1.0, 1.0]. Unknown nodes raise ValueError;
        unknown event types are ignored (no-op) so the orchestrator can pass
        through events this module doesn't model.
        """
        etype = event.get("type")
        actor = event.get("actor")
        target = event.get("target")
        present: Iterable[str] = event.get("present") or []

        if etype == "accusation":
            # Unsupported accusation: accuser loses trust with the accused.
            if actor is not None and target is not None:
                self._adjust(actor, target, -DELTA_ACCUSATION)
                self._adjust(target, actor, -DELTA_ACCUSATION)

        elif etype == "accusation_backed":
            # Evidence-backed accusation lands: the accused loses trust broadly.
            if target is not None:
                bystanders = set(present) if present else set(NODES)
                for p in bystanders:
                    if p != target:
                        self._adjust(p, target, -DELTA_ACCUSATION_BACKED)

        elif etype in ("promise_kept", "promise_broken"):
            # Trust toward the promise-maker (actor, typically the player) moves.
            sign = 1.0 if etype == "promise_kept" else -1.0
            if actor is not None:
                if present:
                    recipients = [p for p in present if p != actor]
                elif target is not None:
                    recipients = [target]
                else:
                    recipients = []
                for r in recipients:
                    self._adjust(r, actor, sign * DELTA_PROMISE)

        elif etype == "defense":
            # One character seen defending/covering for another: mutual bump.
            if actor is not None and target is not None:
                self._adjust(actor, target, DELTA_DEFENSE)
                self._adjust(target, actor, DELTA_DEFENSE)

        # Any other type: intentionally a no-op.

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        """Return the full nested trust map for the GM/judge visualization.

        Shape: ``{from_id: {to_id: score}}`` for every ordered pair of nodes
        (no self-edges). Values are plain floats. Returns a deep copy so callers
        can't mutate internal state.
        """
        return {a: dict(targets) for a, targets in self._trust.items()}


if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # Smoke test (this module only -- no end-to-end, no LLM).
    # ---------------------------------------------------------------------
    tg = TrustGraph()

    def _show(a: str, b: str) -> None:
        print(f"  {a:>7} -> {b:<7} {tg.score(a, b):+.2f}  [{tg.band(a, b)}]")

    print("Initial bands for key pairs:")
    _show("hettie", "slate")   # expect close (0.90)
    _show("slate", "hettie")   # expect close (0.90)
    _show("hettie", "wren")    # expect close (0.80)
    _show("player", "marrow")  # expect neutral (0.00)
    _show("wren", "barnaby")   # expect wary (-0.30)
    _show("marrow", "odette")  # expect neutral (0.10)

    # Sanity checks on the specifically-requested pairs.
    assert tg.band("hettie", "slate") in ("close", "warm"), tg.band("hettie", "slate")
    assert tg.band("player", "marrow") == "neutral", tg.band("player", "marrow")

    print("\nOne directive line example:")
    print("  " + tg.directive("hettie", "wren"))

    print("\nEvent 1: player makes an UNSUPPORTED accusation against Slate.")
    before = tg.band("player", "slate"), tg.band("slate", "player")
    tg.apply_event({"type": "accusation", "actor": "player", "target": "slate"})
    after = tg.band("player", "slate"), tg.band("slate", "player")
    print(f"  player->slate: {before[0]} -> {after[0]} ({tg.score('player','slate'):+.2f})")
    print(f"  slate->player: {before[1]} -> {after[1]} ({tg.score('slate','player'):+.2f})")

    print("\nEvent 2: player KEEPS a promise made to Wren.")
    before_b = tg.band("wren", "player")
    tg.apply_event({"type": "promise_kept", "actor": "player", "target": "wren"})
    after_b = tg.band("wren", "player")
    print(f"  wren->player: {before_b} -> {after_b} ({tg.score('wren','player'):+.2f})")

    print("\nEvent 3: Hettie is seen DEFENDING Wren in a scene.")
    before_c = tg.score("hettie", "wren"), tg.score("wren", "hettie")
    tg.apply_event({"type": "defense", "actor": "hettie", "target": "wren"})
    print(
        f"  hettie->wren: {before_c[0]:+.2f} -> {tg.score('hettie','wren'):+.2f} "
        f"[{tg.band('hettie','wren')}]"
    )
    print(
        f"  wren->hettie: {before_c[1]:+.2f} -> {tg.score('wren','hettie'):+.2f} "
        f"[{tg.band('wren','hettie')}]"
    )

    print("\nEvent 4: evidence-backed accusation lands on Marrow (all present).")
    tg.apply_event({
        "type": "accusation_backed",
        "actor": "player",
        "target": "marrow",
        "present": ["player", "slate", "wren", "hettie"],
    })
    for observer in ("slate", "wren", "hettie"):
        print(
            f"  {observer}->marrow: {tg.score(observer,'marrow'):+.2f} "
            f"[{tg.band(observer,'marrow')}]"
        )

    print("\nto_dict() node count:", len(tg.to_dict()))
    print("Smoke test OK.")
