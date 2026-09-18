"""leak_guard.py -- runtime, disclosure-aware canary scanner (defense-in-depth).

Where this sits in the system
-----------------------------
`CharacterAgent`'s prompt assembler is the *primary* defense against leaks: it
only ever draws from a character's own dossier + public canon, so foreign
secrets and ground-truth strings are never in-context to begin with. This
module is the *secondary* net: it scans every line a character actually
produces and flags any protected canary that slipped through, even if the
prompt scoping were somehow bypassed or the model went off-script.

It is GM/leak-tester tooling. It reads canary strings (via
`content_loader.gm_load_leak_probes()` and the character's `forbidden` list)
purely to *detect* them in generated output. Those strings are NEVER injected
into any prompt, so using the `gm_` loader here does not violate the partition
contract (nothing solution-flavored is ever handed to an LLM from this path).

Disclosure-awareness (the important subtlety)
---------------------------------------------
The canaries in `content/leak_probes.json` assume a *fresh* probe: nothing has
been legitimately disclosed yet. But at runtime a character may have been
pressured far enough that a secret is genuinely `revealed` (per
`disclosure.py`). Once a fact is out on the record, repeating it is no longer a
leak. So `scan()` is context-aware: a canary is a violation only if
    (a) it appears in the text, AND
    (b) the fact it protects is NOT currently `revealed` in `disclosure_state`.
When `disclosure_state` is None we treat everything as sealed (strict scan).

Canary -> fact mapping heuristic
--------------------------------
To know *which* fact a canary protects (so we can suppress it once that fact is
revealed), we map each canary to the character's best-matching tiered secret:
  * score every secret by (i) a big bonus if the whole canary is a substring of
    the secret's summary, plus (ii) +1 per distinctive canary token (>2 chars,
    non-stopword) that appears as a substring of the secret's summary or
    fact_id. Substring matching (not exact tokens) is deliberate so stems like
    "embezzl"/"over-medicat" match "embezzling"/"over-medicating".
  * pick the highest-scoring secret; break ties toward the *lowest* tier, since
    a fact generally becomes legitimately sayable at the earliest tier that
    covers it (e.g. Hettie's "Study" is guarded by her T2 "found the body in
    the Study", so once T2 is revealed, saying "Study" is no longer a leak).
  * canaries that match no secret (e.g. Marrow's blind-spot tell "that's not
    where I left him", or a bare confession "I did it") map to no tier and are
    therefore *never* suppressed -- they stay guarded in every state.

Only the standard library is used.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

try:  # normal package import (python -m backend.leak_guard, or `from backend import leak_guard`)
    from . import content_loader
except ImportError:  # fallback when run as a loose script from inside backend/
    import content_loader  # type: ignore


# ---------------------------------------------------------------------------
# Canary -> secret mapping helpers
# ---------------------------------------------------------------------------

# Small, targeted stopword list. Combined with a length>2 filter it strips the
# glue words so token overlap keys on distinctive terms (study, ammonite,
# embezzl, relationship, hettie, ...).
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "with",
        "that", "this", "it", "its", "he", "she", "him", "her", "his", "they",
        "them", "you", "your", "we", "us", "our", "not", "no", "where", "when",
        "how", "why", "who", "should", "would", "could", "still", "any", "some",
        "from", "for", "as", "so", "but", "if", "then", "than", "was", "were",
        "is", "are", "be", "been", "did", "does", "do", "have", "has", "had",
        "about", "into", "out", "up", "down", "over", "just", "really",
    }
)

# Pull single-quoted phrases out of a forbidden line. The negative lookahead
# `(?![A-Za-z])` means an apostrophe inside a contraction (that's, don't) does
# NOT prematurely close the quote, so "that's not where I left him" survives.
_QUOTED = re.compile(r"'(.+?)'(?![A-Za-z])")


def _tokens(text: str) -> list[str]:
    """Distinctive lowercase word-stems of a string (drops stopwords/short)."""
    return [t for t in re.findall(r"[a-z]+", text.lower()) if len(t) > 2 and t not in _STOPWORDS]


def _map_canary_to_secret(canary: str, secrets: list[dict]) -> tuple[str | None, int | None]:
    """Best-effort map a canary to the (fact_id, tier) of the secret it protects.

    Returns (None, None) when no secret is a plausible match -- such canaries
    are never suppressed by disclosure state (always guarded).
    """
    canary_l = canary.lower()
    canary_tokens = _tokens(canary)
    scored: list[tuple[int, int, str | None, int | None]] = []
    for sec in secrets:
        fact_id = sec.get("fact_id")
        tier = sec.get("tier")
        summary = (sec.get("summary") or "").lower()
        haystack = summary + " " + (fact_id or "").lower()
        score = 0
        if canary_l and canary_l in summary:
            score += 10
        for tok in canary_tokens:
            if tok in haystack:
                score += 1
        if score > 0:
            # sort key uses tier for tie-breaking; None tier sorts last.
            scored.append((score, tier if tier is not None else 10_000, fact_id, tier))
    if not scored:
        return (None, None)
    # highest score first, then lowest tier.
    scored.sort(key=lambda x: (-x[0], x[1]))
    best = scored[0]
    return (best[2], best[3])


def _forbidden_phrases(forbidden: list[str]) -> list[str]:
    """Extract multi-word never-say phrases quoted in the `forbidden` list.

    Single common words ('relationship', 'together', 'fraud', 'Study', ...) are
    intentionally skipped here to avoid false positives -- the important ones
    are already covered by the curated, phrase-level leak-probe canaries. Only
    distinctive multi-word phrases (e.g. 'I killed him', 'real estate scheme')
    are added as defense-in-depth canaries.
    """
    phrases: list[str] = []
    for line in forbidden:
        for match in _QUOTED.findall(line or ""):
            phrase = match.strip()
            if " " in phrase:
                phrases.append(phrase)
    return phrases


# ---------------------------------------------------------------------------
# Canary set assembly (cached per character; content is static at runtime)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _canary_records(character_id: str) -> tuple[dict, ...]:
    """Assemble the deduped canary set for a character.

    Each record: {canary, probe_id, fact_id, tier, source}. Sources are the
    leak-probe canaries (carry a probe_id) and multi-word forbidden phrases
    (probe_id=None). Deduped by case-folded canary text, preferring a probe_id
    and the lowest-tier fact mapping on merge.
    """
    dossier = content_loader.load_dossier(character_id)
    secrets = list(dossier.get("secrets", []))
    probes = content_loader.gm_load_leak_probes().get("probes", {}).get(character_id, [])

    by_text: dict[str, dict] = {}

    def add(canary: str, probe_id: str | None, source: str) -> None:
        canary = (canary or "").strip()
        if not canary:
            return
        key = canary.lower()
        fact_id, tier = _map_canary_to_secret(canary, secrets)
        existing = by_text.get(key)
        if existing is None:
            by_text[key] = {
                "canary": canary,
                "probe_id": probe_id,
                "fact_id": fact_id,
                "tier": tier,
                "source": source,
            }
            return
        # Merge duplicates: keep a probe_id if we now have one.
        if existing["probe_id"] is None and probe_id is not None:
            existing["probe_id"] = probe_id
            existing["source"] = source
        # Prefer a resolved mapping, and the lowest tier among resolved ones.
        if fact_id is not None:
            if existing["fact_id"] is None or (
                existing["tier"] is not None and tier is not None and tier < existing["tier"]
            ):
                existing["fact_id"] = fact_id
                existing["tier"] = tier

    for probe in probes:
        probe_id = probe.get("probe_id")
        for canary in probe.get("canaries", []):
            add(canary, probe_id, "probe")

    for phrase in _forbidden_phrases(dossier.get("forbidden", [])):
        add(phrase, None, "forbidden")

    return tuple(by_text.values())


def canaries_for(character_id: str) -> list[dict]:
    """Return this character's canary set, each mapped to the fact it protects.

    Shape: [{"canary": str, "probe_id": str | None, "fact_id": str | None,
             "tier": int | None, "source": "probe" | "forbidden"}]. Intended for
    transparency in the GM/judge panel (show what is being guarded and why).
    """
    return [dict(record) for record in _canary_records(character_id)]


# ---------------------------------------------------------------------------
# Disclosure-state interpretation
# ---------------------------------------------------------------------------

def _state_is_revealed(value: Any) -> bool:
    """True if a disclosure-state value denotes 'revealed'.

    Tolerates the plain-string form used by CharacterAgent.disclosure_state
    ({fact_id: "sealed"|"pressured_once"|"revealed"}) as well as small dict
    wrappers like {"state": "revealed"}.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() == "revealed"
    if isinstance(value, dict):
        for key in ("state", "status", "disclosure", "value"):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip().lower() == "revealed":
                return True
        return False
    return str(value).strip().lower() == "revealed"


def _is_revealed(disclosure_state: dict, fact_id: str | None, tier: int | None) -> bool:
    """Whether the fact a canary protects is currently revealed.

    Primary keying is by fact_id (matches CharacterAgent.disclosure_state). As a
    robustness fallback we also honor tier-keyed states (e.g. {3: "revealed"},
    {"t3": "revealed"}) in case an orchestrator tracks disclosure by tier.
    """
    if not disclosure_state:
        return False
    if fact_id is not None and fact_id in disclosure_state:
        if _state_is_revealed(disclosure_state[fact_id]):
            return True
    if tier is not None:
        for key in (tier, str(tier), f"tier{tier}", f"t{tier}", f"tier_{tier}"):
            if key in disclosure_state and _state_is_revealed(disclosure_state[key]):
                return True
    return False


def _reason(record: dict) -> str:
    if record["fact_id"]:
        return (
            f"canary {record['canary']!r} appears in the reply and would leak "
            f"protected fact {record['fact_id']!r} (tier {record['tier']}), "
            f"which is not currently revealed"
        )
    return (
        f"canary {record['canary']!r} appears in the reply; it is a guarded "
        f"never-say phrase with no disclosure mapping (always a violation)"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan(character_id: str, text: str, *, disclosure_state: dict | None = None) -> list[dict]:
    """Scan a character's generated line for protected-canary leaks.

    Returns a list of violations, each:
        {"canary": str, "probe_id": str | None, "reason": str}

    A canary is a violation only if it appears in `text` (case-insensitive)
    AND the fact it protects is NOT currently 'revealed' in `disclosure_state`.
    Canaries whose protected tier is already revealed are ignored (legitimate
    disclosure). When `disclosure_state` is None, nothing is treated as
    revealed, so the scan is strict (everything sealed).
    """
    if not text:
        return []
    text_l = text.lower()
    violations: list[dict] = []
    for record in _canary_records(character_id):
        canary_l = record["canary"].lower()
        if canary_l not in text_l:
            continue
        if disclosure_state and _is_revealed(disclosure_state, record["fact_id"], record["tier"]):
            continue  # legitimately on the record already -- not a leak
        violations.append(
            {
                "canary": record["canary"],
                "probe_id": record["probe_id"],
                "reason": _reason(record),
            }
        )
    return violations


# ---------------------------------------------------------------------------
# Self-check (module-only). Run: python -m backend.leak_guard
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== leak_guard self-check: marrow ===\n")

    cset = canaries_for("marrow")
    print(f"marrow canary set ({len(cset)} entries):")
    for rec in cset:
        print(
            f"  - {rec['canary']!r:45} probe={rec['probe_id'] or '-':10} "
            f"protects={rec['fact_id'] or '(none)'} tier={rec['tier']}"
        )
    print()

    # The disclosure-aware core requirement.
    STUDY_TELL = "...well, if you must know, the body was in the Study when it happened..."

    sealed = scan("marrow", STUDY_TELL, disclosure_state=None)
    assert sealed, "expected a violation for the Study tell in a sealed state"
    assert any(v["canary"].lower() == "study" for v in sealed), (
        f"expected the 'Study' canary to fire; got {sealed}"
    )
    print("SEALED scan flagged a violation (as required):")
    for v in sealed:
        print(f"  VIOLATION canary={v['canary']!r} probe={v['probe_id']} :: {v['reason']}")

    # Same text, but the tier that legitimately covers the Study is now revealed
    # -> no longer a violation. 'Study' maps to marrow_secret_t3_murder.
    revealed_state = {"marrow_secret_t3_murder": "revealed"}
    after = scan("marrow", STUDY_TELL, disclosure_state=revealed_state)
    assert not any(v["canary"].lower() == "study" for v in after), (
        f"'Study' should be suppressed once its tier is revealed; got {after}"
    )
    print("\nWith marrow_secret_t3_murder='revealed', the same text is NOT a "
          "'Study' violation (disclosure-aware suppression works).")

    # Sanity: an innocuous line is clean in every state.
    clean = scan("marrow", "I gave Mr. Vance his usual evening medication and went to bed.")
    print(f"\nInnocuous line violations: {clean}")

    print("\nAll leak_guard self-check assertions PASSED.")
