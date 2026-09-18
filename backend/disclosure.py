"""Tiered disclosure ladder for the Nautilus House game.

This module implements the "disclosure ladder" described in section 3 of the
design plan. It is a *pure-Python, standard-library-only* module:

  * No pip installs and no LLM calls.
  * It does NOT import `character_agent.py` or `trust_graph.py` (those are
    built in parallel). Instead it receives their relevant outputs -- a trust
    band string and a goal-stack-inversion flag -- as plain parameters.

Role in the system
-------------------
Each suspect gets one :class:`DisclosureTracker`. It tracks, per secret tier,
a state in {"sealed", "pressured_once", "revealed"}. The orchestrator feeds it
pressure (what was just said, who's in the room, what evidence was named, etc.)
and it decides *when* a tier's secret becomes legitimately disclosable. It does
NOT write the character's line -- it only computes permission. The orchestrator
passes :meth:`DisclosureTracker.allowed_facts` into
``CharacterAgent.respond(..., allowed_facts=...)`` so the LLM may reveal a fact
naturally rather than reciting a scripted string.

Leak-guard contract
--------------------
Once a tier is "revealed" here, its ``fact_id`` is on the list returned by
:meth:`allowed_facts`. Downstream ``leak_guard.py`` should treat any fact whose
tier is "revealed" as *legitimately disclosable* -- i.e. its appearance in a
character's line is NOT a leak-probe violation. A `leak_block` secret is only a
violation while its tier is still "sealed"/"pressured_once".

Heuristic-matching disclaimer
-----------------------------
Evidence detection here is deliberately simple: keyword/substring overlap
between the incoming text (utterance + any named clues) and each secret's
`summary` from the dossier, plus the matching clue text from `clues.json`.
This is a heuristic and can be upgraded later (e.g. embeddings, an explicit
clue->tier mapping, or an NLI "denial-incoherent" check) WITHOUT changing the
public interface of :class:`DisclosureTracker`.
"""

from __future__ import annotations

import re
from typing import Any

# `content_loader` lives in the same `backend/` package. Support both running
# as part of the package (`from backend import disclosure`) and running this
# file directly as a script (`python backend/disclosure.py`).
try:  # package import
    from . import content_loader
except ImportError:  # running as a script; backend/ is on sys.path[0]
    import content_loader  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Tunable constants (documented so a later phase can adjust behavior easily).
# ---------------------------------------------------------------------------

SEALED = "sealed"
PRESSURED_ONCE = "pressured_once"
REVEALED = "revealed"

# Trust bands (from trust_graph.py's band translation), mapped to an ordinal.
# Higher trust in the requester => easier disclosure (lower thresholds).
_TRUST_ORDINAL: dict[str, int] = {
    "hostile": -2,
    "wary": -1,
    "neutral": 0,
    "warm": 1,
    "close": 2,
}

# Base number of *distinct* direct questions on a topic needed to clear T1.
_BASE_T1_QUESTION_THRESHOLD = 2

# Base number of overlapping salient keywords needed to count as "evidence".
_BASE_EVIDENCE_KEYWORD_THRESHOLD = 2

# Characters whose Tier-3 secret is IMMUNE to evidence / accusation pressure and
# unlocks ONLY on value pressure (goal-stack inversion). See the module-level
# note about Hettie: a plain murder accusation must never unlock her T3 ("she
# moved the body"); it opens only once Wren is provably innocent
# (`goal_stack_inverted=True`), which collapses her protect-Wren goal. A later
# phase flips that flag; encoding it here keeps this deterministic.
_T3_VALUE_PRESSURE_ONLY: frozenset[str] = frozenset({"hettie"})

# Very small stopword set for keyword extraction -- intentionally minimal so the
# matching stays a transparent heuristic.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "that", "this", "with", "was", "wasn", "were",
        "she", "her", "hers", "him", "his", "them", "they", "you", "your",
        "not", "but", "had", "has", "have", "did", "does", "done", "are",
        "who", "what", "when", "where", "why", "how", "which", "into", "from",
        "out", "about", "some", "any", "all", "one", "two", "been", "being",
        "its", "our", "their", "there", "here", "then", "than", "just",
        "get", "got", "can", "cant", "wont", "dont", "isnt", "roughly",
    }
)

# Matches "alphanumeric run" tokens; used with a length filter below.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _keywords(text: str) -> set[str]:
    """Extract lowercased salient keyword tokens from free text.

    Heuristic: alphanumeric runs of length >= 3 that aren't stopwords. Kept
    deliberately simple/transparent -- see the module docstring's disclaimer.
    """
    if not text:
        return set()
    return {
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) >= 3 and tok not in _STOPWORDS
    }


class DisclosureTracker:
    """Tracks, for one character, the sealed/pressured/revealed state per tier.

    Public interface (this is the contract the orchestrator wires against):

        __init__(self, character_id: str)
        register_pressure(self, *, utterance, asked_topics=None,
                          present_accusers=None, clues_named=None,
                          trade_offered=False, trust_band="neutral",
                          goal_stack_inverted=False) -> dict
        allowed_facts(self) -> list
        state(self) -> dict
    """

    def __init__(self, character_id: str) -> None:
        # Loads this character's own dossier via the AGENT-FACING loader only.
        # We never touch ground_truth / leak_probes here.
        self.character_id = character_id
        dossier = content_loader.load_dossier(character_id)

        # secrets: list of {fact_id, tier, summary, leak_block}. Index by tier.
        self._secrets_by_tier: dict[int, dict[str, Any]] = {}
        for secret in dossier.get("secrets", []):
            self._secrets_by_tier[int(secret["tier"])] = secret

        # Every tier starts sealed.
        self._tier_state: dict[int, str] = {
            tier: SEALED for tier in self._secrets_by_tier
        }

        # Cumulative count of distinct direct questions seen per tier topic.
        self._question_counts: dict[int, int] = {
            tier: 0 for tier in self._secrets_by_tier
        }

        # Precompute salient keywords for each tier's secret summary.
        self._secret_keywords: dict[int, set[str]] = {
            tier: _keywords(secret.get("summary", ""))
            for tier, secret in self._secrets_by_tier.items()
        }

        # Precompute (clue_id -> keywords) and (clue_id -> statement) from the
        # player-facing clue board so `clues_named` can be resolved to real
        # evidence text. Anchor + supporting clues are both public.
        self._clue_keywords: dict[str, set[str]] = {}
        self._clue_text: dict[str, str] = {}
        clues = content_loader.load_clues()
        for group in ("anchor_clues", "supporting_clues"):
            for clue in clues.get(group, []):
                clue_id = clue.get("clue_id", "")
                text = " ".join(
                    [clue.get("statement", "")] + list(clue.get("proves", []))
                )
                self._clue_text[clue_id] = text
                self._clue_keywords[clue_id] = _keywords(text)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _resolve_clue_text(self, clue_ref: str) -> str:
        """Resolve a `clues_named` entry to evidence text.

        Accepts either a known clue_id (resolved from clues.json) or a raw
        free-text description of the evidence (used as-is).
        """
        if clue_ref in self._clue_text:
            return self._clue_text[clue_ref]
        return clue_ref

    def _evidence_strength(self, incoming_kw: set[str], tier: int) -> int:
        """Number of salient keywords shared between incoming text and a tier."""
        return len(incoming_kw & self._secret_keywords.get(tier, set()))

    def _tier_for_topic(self, topic: str) -> int | None:
        """Map a free-text `asked_topics` entry to the tier it's about.

        Returns the tier with the most keyword overlap (>=1), else None.
        """
        topic_kw = _keywords(topic)
        if not topic_kw:
            return None
        best_tier: int | None = None
        best_overlap = 0
        for tier, secret_kw in self._secret_keywords.items():
            overlap = len(topic_kw & secret_kw)
            if overlap > best_overlap:
                best_overlap = overlap
                best_tier = tier
        return best_tier if best_overlap >= 1 else None

    def _promote(self, tier: int, new_state: str) -> bool:
        """Move a tier forward in the ladder; never downgrade. Returns changed."""
        order = {SEALED: 0, PRESSURED_ONCE: 1, REVEALED: 2}
        if order[new_state] > order[self._tier_state[tier]]:
            self._tier_state[tier] = new_state
            return True
        return False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def register_pressure(
        self,
        *,
        utterance: str,
        asked_topics: list[str] | None = None,
        present_accusers: list[str] | None = None,
        clues_named: list[str] | None = None,
        trade_offered: bool = False,
        trust_band: str = "neutral",
        goal_stack_inverted: bool = False,
    ) -> dict:
        """Register one turn of pressure and update tier states.

        Computes the four pressure signals from the plan and applies the
        tiered unlock rules, with trust lowering the thresholds.

        Parameters
        ----------
        utterance:
            The incoming line (from the player or another character).
        asked_topics:
            Free-text topics the requester is asking about this turn. Two or
            more distinct questions on a tier's topic (cumulatively) can clear
            that tier's question threshold.
        present_accusers:
            Other characters accusing this character *in-scene*. Non-empty =>
            social pressure.
        clues_named:
            Evidence explicitly presented -- clue_ids (resolved against
            clues.json) or raw descriptions. Contributes evidence pressure.
        trade_offered:
            The requester offered protection/silence/a secret => trade pressure.
        trust_band:
            The requester's trust band toward this character
            ("hostile"/"wary"/"neutral"/"warm"/"close"). Higher trust lowers
            thresholds; "hostile" raises them (stonewalling).
        goal_stack_inverted:
            A core goal has collapsed (value pressure) -- e.g. Hettie's
            protect-Wren goal once Wren is provably innocent.

        Returns
        -------
        dict with:
            "transitions": {tier: {"from": ..., "to": ...}} for tiers that
                changed state on THIS call,
            "allowed_facts": the fact_ids now permitted to reveal (all
                currently-revealed tiers),
            "signals": the four computed pressure signals (for GM/debug).
        """
        asked_topics = asked_topics or []
        present_accusers = present_accusers or []
        clues_named = clues_named or []

        trust_ord = _TRUST_ORDINAL.get(trust_band, 0)

        # --- Trust-adjusted thresholds (higher trust => lower thresholds). ---
        q_threshold = max(1, _BASE_T1_QUESTION_THRESHOLD - trust_ord)
        # Clamp evidence keyword threshold to a sane [1, 3] range.
        evidence_threshold = max(
            1, min(3, _BASE_EVIDENCE_KEYWORD_THRESHOLD - trust_ord)
        )

        # --- Build the incoming keyword corpus (utterance + named clues). ---
        incoming_parts = [utterance] + [
            self._resolve_clue_text(c) for c in clues_named
        ]
        incoming_kw = _keywords(" ".join(incoming_parts))

        # --- Four pressure signals from the plan. -------------------------
        # 1) Evidence pressure, per tier (keyword overlap >= threshold).
        evidence_present: dict[int, bool] = {
            tier: self._evidence_strength(incoming_kw, tier) >= evidence_threshold
            for tier in self._secrets_by_tier
        }
        any_partial_evidence: dict[int, bool] = {
            tier: self._evidence_strength(incoming_kw, tier) >= 1
            for tier in self._secrets_by_tier
        }
        # 2) Social pressure -- an in-scene accusation.
        social_pressure = bool(present_accusers)
        # 3) Trade pressure.
        trade_pressure = bool(trade_offered)
        # 4) Value pressure -- goal-stack inversion.
        value_pressure = bool(goal_stack_inverted)

        # --- Fold in this turn's direct questions per tier. ---------------
        questioned_tiers: set[int] = set()
        for topic in asked_topics:
            tier = self._tier_for_topic(topic)
            if tier is not None:
                self._question_counts[tier] += 1
                questioned_tiers.add(tier)

        # --- Snapshot prior state so we can report transitions. -----------
        prior_state = dict(self._tier_state)

        # --- Apply unlock rules, lowest tier first. -----------------------
        for tier in sorted(self._secrets_by_tier):
            revealed = self._compute_reveal(
                tier=tier,
                evidence_present=evidence_present,
                social_pressure=social_pressure,
                trade_pressure=trade_pressure,
                value_pressure=value_pressure,
                q_threshold=q_threshold,
            )
            if revealed:
                self._promote(tier, REVEALED)
                continue

            # Not enough to reveal -- record that pressure was at least felt.
            felt_pressure = (
                tier in questioned_tiers
                or any_partial_evidence[tier]
                or social_pressure
                or trade_pressure
            )
            if felt_pressure:
                self._promote(tier, PRESSURED_ONCE)

        # --- Assemble return payload. -------------------------------------
        transitions: dict[int, dict[str, str]] = {}
        for tier in sorted(self._secrets_by_tier):
            if self._tier_state[tier] != prior_state[tier]:
                transitions[tier] = {
                    "from": prior_state[tier],
                    "to": self._tier_state[tier],
                }

        return {
            "transitions": transitions,
            "allowed_facts": self.allowed_facts(),
            "signals": {
                "evidence_present": evidence_present,
                "social_pressure": social_pressure,
                "trade_pressure": trade_pressure,
                "value_pressure": value_pressure,
                "question_counts": dict(self._question_counts),
            },
        }

    def _compute_reveal(
        self,
        *,
        tier: int,
        evidence_present: dict[int, bool],
        social_pressure: bool,
        trade_pressure: bool,
        value_pressure: bool,
        q_threshold: int,
    ) -> bool:
        """Return True if `tier` meets its unlock rule (given current counters).

        Unlock rules (section 3 of the plan):
          * T1: 2+ direct questions on the topic OR any evidence pressure.
          * T2: named evidence OR a trade.
          * T3: denial-incoherent evidence OR goal-stack inversion.
                (Note: a plain accusation -- social pressure -- never unlocks
                 T3 on its own, which is what makes Hettie immune to bare
                 murder accusations.)

        Special case: for characters in `_T3_VALUE_PRESSURE_ONLY` (Hettie),
        T3 unlocks ONLY on value pressure -- evidence never does.
        """
        if tier == 1:
            return (
                self._question_counts.get(1, 0) >= q_threshold
                or evidence_present.get(1, False)
            )

        if tier == 2:
            # "named evidence or a trade". Keyword-matched evidence over the
            # clue board + secret summary is our heuristic for "named evidence".
            return evidence_present.get(2, False) or trade_pressure

        if tier == 3:
            if self.character_id in _T3_VALUE_PRESSURE_ONLY:
                # Immune to evidence/accusation -- value pressure is the only key.
                return value_pressure
            # General case: goal-stack inversion, OR "denial-incoherent"
            # evidence -- modeled as fresh T3-topic evidence once the T2 secret
            # is already out, so continued denial no longer holds together.
            denial_incoherent = (
                evidence_present.get(3, False)
                and self._tier_state.get(2) == REVEALED
            )
            return value_pressure or denial_incoherent

        # Unknown tier: never auto-reveal.
        return False

    def allowed_facts(self) -> list:
        """Return fact_ids whose tier is currently "revealed".

        The character MAY disclose these if asked. Orchestrator passes this
        into ``CharacterAgent.respond(..., allowed_facts=...)``. Downstream
        leak-guard should treat these as legitimately disclosable.
        """
        return [
            self._secrets_by_tier[tier]["fact_id"]
            for tier in sorted(self._secrets_by_tier)
            if self._tier_state[tier] == REVEALED
        ]

    def state(self) -> dict:
        """Return the current per-tier states for the GM/judge memory panel."""
        return {
            "character_id": self.character_id,
            "tiers": {
                str(tier): {
                    "fact_id": self._secrets_by_tier[tier]["fact_id"],
                    "state": self._tier_state[tier],
                    "leak_block": self._secrets_by_tier[tier].get(
                        "leak_block", False
                    ),
                    "questions_seen": self._question_counts[tier],
                }
                for tier in sorted(self._secrets_by_tier)
            },
            "allowed_facts": self.allowed_facts(),
        }


# ==========================================================================
# Module-only smoke test (no end-to-end, no LLM). Run: python backend/disclosure.py
# ==========================================================================
if __name__ == "__main__":

    def _print_call(label: str, tracker: DisclosureTracker, result: dict) -> None:
        transitions = result["transitions"]
        if transitions:
            parts = [
                f"T{tier}: {t['from']} -> {t['to']}"
                for tier, t in sorted(transitions.items())
            ]
            trans_str = "; ".join(parts)
        else:
            trans_str = "(no transitions)"
        print(f"  {label}")
        print(f"    transitions: {trans_str}")
        print(f"    allowed_facts: {result['allowed_facts']}")

    print("=" * 74)
    print("SLATE -- camera-cut secret unlocking under evidence pressure")
    print("=" * 74)
    slate = DisclosureTracker("slate")
    print(f"  initial state: "
          f"{ {k: v['state'] for k, v in slate.state()['tiers'].items()} }")

    # 1) Ask directly about the corridor camera footage. This already carries
    #    enough overlap with the T1 "feed gap" secret to register as evidence
    #    pressure, so T1 unlocks here (heuristic matching -- see disclaimer).
    r = slate.register_pressure(
        utterance="What happened with the Corridor camera footage that night?",
        asked_topics=["corridor camera footage that night"],
    )
    _print_call("Ask about the corridor footage -> expect T1 reveal", slate, r)

    # 2) Present the camera-gap clue as hard evidence (reinforces T1).
    r = slate.register_pressure(
        utterance="The security log shows a 35-minute gap in the Corridor "
        "camera footage that night.",
        clues_named=["clue_camera_gap"],
    )
    _print_call("Present camera-gap evidence (T1 already open)", slate, r)

    # 3) Confront her that she personally cut the feed -> T2 (she cut it).
    r = slate.register_pressure(
        utterance="You personally cut the camera feed yourself -- that was no "
        "malfunction.",
    )
    _print_call("Confront 'you cut the feed' -> expect T2 reveal", slate, r)

    print(f"  final slate allowed_facts: {slate.allowed_facts()}")
    assert "slate_secret_t1_feed_gap" in slate.allowed_facts(), "T1 should unlock"
    assert "slate_secret_t2_she_cut_it" in slate.allowed_facts(), "T2 should unlock"
    # T3 (the relationship) should still be sealed -- no relationship evidence.
    assert "slate_secret_t3_why" not in slate.allowed_facts(), "T3 must stay sealed"
    print("  [OK] Slate T1+T2 unlocked under evidence pressure; T3 still sealed.")

    print()
    print("=" * 74)
    print("HETTIE -- T3 stays sealed under accusation, unlocks on goal inversion")
    print("=" * 74)
    hettie = DisclosureTracker("hettie")
    print(f"  initial state: "
          f"{ {k: v['state'] for k, v in hettie.state()['tiers'].items()} }")

    # 1) Odette flatly accuses Hettie of the murder -- social pressure only.
    r = hettie.register_pressure(
        utterance="Everyone knows you did it, Hettie -- you killed Vance.",
        present_accusers=["odette"],
    )
    _print_call("Bare murder accusation -> T3 must NOT reveal", hettie, r)
    assert "hettie_secret_t3_why_wren" not in hettie.allowed_facts(), (
        "Hettie's T3 must not unlock on a plain accusation"
    )

    # 2) Even hard evidence she was in the Study must not open T3 (immune).
    r = hettie.register_pressure(
        utterance="There's Study display-case wax on your apron -- you were in "
        "the Study and you moved the body.",
        clues_named=["clue_wax_trace"],
        present_accusers=["odette", "barnaby"],
    )
    _print_call("Evidence she moved the body -> T3 still sealed", hettie, r)
    assert "hettie_secret_t3_why_wren" not in hettie.allowed_facts(), (
        "Hettie's T3 must not unlock on evidence/accusation"
    )

    # 3) Wren is proven innocent -> goal-stack inversion -> T3 unlocks.
    r = hettie.register_pressure(
        utterance="Wren has an airtight alibi -- she was never near the Study. "
        "She's innocent.",
        goal_stack_inverted=True,
    )
    _print_call("goal_stack_inverted=True -> expect T3 reveal", hettie, r)
    assert "hettie_secret_t3_why_wren" in hettie.allowed_facts(), (
        "Hettie's T3 must unlock on goal-stack inversion"
    )
    print("  [OK] Hettie T3 sealed under accusation/evidence; opened on "
          "goal-stack inversion.")

    print()
    print("Final GM state panels:")
    for tracker in (slate, hettie):
        st = tracker.state()
        tiers = {k: v["state"] for k, v in st["tiers"].items()}
        print(f"  {st['character_id']}: {tiers}")
    print()
    print("All disclosure smoke-test assertions passed.")
