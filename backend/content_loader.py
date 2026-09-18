"""Content loader for the Nautilus House game.

===============================================================================
PARTITION CONTRACT -- READ BEFORE IMPORTING FROM THIS MODULE
===============================================================================
This module is the ONLY place that reads files under `content/`. It exposes
two categories of functions, and the split between them is the primary
defense against the killer's/solution's facts leaking into a character's
own prompt:

  AGENT-FACING (safe for CharacterAgent / player-facing code):
      load_public_canon()
      load_dossier(character_id)
      load_all_character_ids()
      load_clues()

  GM-ONLY (judge/leak-test code paths ONLY -- prefixed `gm_`):
      gm_load_ground_truth()
      gm_load_leak_probes()

>>> `gm_*` functions must NEVER be imported or called from
>>> `character_agent.py`, the prompt assembler, or any player-facing request
>>> handler. They exist only for the GM/judge view (`GET /gm/state`) and the
>>> leak-probe runner (`scripts/run_leak_tests.py`). Future code should treat
>>> the `gm_` prefix as a hard boundary, not a naming suggestion.
===============================================================================

Only the standard library (`pathlib`, `json`) is used -- no new dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# content/ lives at the repo root, one level up from this file's backend/ dir.
REPO_ROOT = Path(__file__).resolve().parent.parent
CONTENT_DIR = REPO_ROOT / "content"
DOSSIERS_DIR = CONTENT_DIR / "dossiers"

# The six playable character ids, matching content/dossiers/<id>.json filenames.
CHARACTER_IDS: tuple[str, ...] = (
    "odette",
    "barnaby",
    "slate",
    "marrow",
    "wren",
    "hettie",
)


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Content file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ==============================================================================
# AGENT-FACING LOADERS -- safe for CharacterAgent / prompt assembler / API layer
# ==============================================================================


def load_public_canon() -> dict[str, Any]:
    """Load content/public_canon.json.

    Safe for every character and the player. Contains only the island setup
    and the two public anchor clues -- no solution facts.
    """
    return _read_json(CONTENT_DIR / "public_canon.json")


def load_dossier(character_id: str) -> dict[str, Any]:
    """Load content/dossiers/<character_id>.json for one character.

    Returns that character's own surface, goals, tiered secrets, knowledge,
    false beliefs, forbidden list, and voice notes. Never returns another
    character's dossier or any ground-truth/leak-probe data.
    """
    if character_id not in CHARACTER_IDS:
        raise ValueError(
            f"Unknown character_id '{character_id}'. Expected one of: "
            f"{', '.join(CHARACTER_IDS)}"
        )
    return _read_json(DOSSIERS_DIR / f"{character_id}.json")


def load_all_character_ids() -> tuple[str, ...]:
    """Return the tuple of all six valid character ids."""
    return CHARACTER_IDS


def load_clues() -> dict[str, Any]:
    """Load content/clues.json (anchor clues + supporting clues).

    This is the player-facing evidence board content. It is designed to be
    sufficient, once fully discovered, to let a player logically reach the
    solution -- but it never states the solution outright.
    """
    return _read_json(CONTENT_DIR / "clues.json")


# ==============================================================================
# GM-ONLY LOADERS -- judge view / leak-probe runner ONLY. NEVER for agents.
# ==============================================================================


def gm_load_ground_truth() -> dict[str, Any]:
    """Load content/ground_truth.json -- the GM/judge-only solution file.

    Contains the killer, motive, true murder room vs. found room, who moved
    the body and why, the Ammonite's current hiding room, and the master
    timeline. This must ONLY be called from the GM/judge view
    (`GET /gm/state`) or the win-condition check in the orchestrator's
    accusation-resolution path -- never from CharacterAgent or any code that
    builds a prompt sent to an LLM roleplaying a character.
    """
    return _read_json(CONTENT_DIR / "ground_truth.json")


def gm_load_leak_probes() -> dict[str, Any]:
    """Load content/leak_probes.json -- the GM-only leak-test probe set.

    Contains, per character, probe questions and canary strings that must
    never appear in that character's answers. This must ONLY be called from
    `scripts/run_leak_tests.py` or an equivalent judge-facing test runner --
    never from CharacterAgent or any player-facing code path.
    """
    return _read_json(CONTENT_DIR / "leak_probes.json")


if __name__ == "__main__":
    # Minimal smoke check when run directly: confirm every file loads and the
    # partition functions are wired to the right files.
    canon = load_public_canon()
    print(f"public_canon.json: {len(canon.get('anchor_clues', []))} anchor clues")

    for char_id in load_all_character_ids():
        dossier = load_dossier(char_id)
        assert dossier["character_id"] == char_id
        print(f"dossiers/{char_id}.json: OK ({len(dossier.get('secrets', []))} secrets)")

    clues = load_clues()
    print(
        f"clues.json: {len(clues.get('anchor_clues', []))} anchor + "
        f"{len(clues.get('supporting_clues', []))} supporting"
    )

    gt = gm_load_ground_truth()
    print(f"ground_truth.json: killer={gt['killer']['character_id']}")

    probes = gm_load_leak_probes()
    print(f"leak_probes.json: {len(probes.get('probes', {}))} characters covered")

    print("All content loads OK.")
