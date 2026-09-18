"""Define, place, and track discoverable clues in a generated house.

Works alongside ``CharClass`` and ``trust_engine`` without depending on
either at import time. A clue is a plain dict:

    {
        "description": str,
        "room": str,            # actual room in this generated house
        "discovered": bool,
        "corroborates": str | None,  # character name this clue backs up
        "contradicts": str | None,   # character name this clue disproves
    }

The intended flow, tying this together with ``trust_engine``:

    1. trust_engine.update_trust() looks at room co-occupancy and marks
       some pairs "weak" -- unresolved, because two people alone together
       can't be told apart from an innocent pair.
    2. The player searches rooms and calls discover() on clues found there.
    3. A discovered clue that "contradicts" or "corroborates" someone gives
       you a concrete, non-testimonial reason to pass into
       trust_engine.resolve_weak_pair() -- see the demo at the bottom.

This module never touches a character's ``trusts`` dict directly. It only
tracks clues; the caller decides how a clue gets applied.
"""

from __future__ import annotations


# Edit this list to add/remove/relocate clues. Each entry is:
# (clue_id, description, preferred_room, corroborates, contradicts)
# ``corroborates``/``contradicts`` should be a character name (matching
# CharClass.CHARACTER_DEFINITIONS) or None.
CLUE_DEFINITIONS = [
    (
        "bloodstain",
        "A faint, hastily-cleaned bloodstain on the Study rug -- the body "
        "was not always wherever it was found.",
        "Study",
        None,
        None,
    ),
    (
        "missing_fossil",
        "The Ammonite fossil is missing from its case in the Study.",
        "Study",
        None,
        None,
    ),
    (
        "muddy_footprints",
        "Muddy footprints leading out through the Mudroom, far heavier and "
        "more dragged-looking than a normal walk.",
        "Mudroom",
        None,
        "Hettie",
    ),
    (
        "forged_letter",
        "A half-burned letter in the fireplace, clearly forged and signed "
        "with a shaky hand.",
        "Game Room",
        None,
        "Odette",
    ),
    (
        "old_ledger",
        "An old financial ledger with entries that don't add up, tucked "
        "behind the bookshelf.",
        "Library",
        None,
        "Barnaby",
    ),
    (
        "camera_log_gap",
        "The security log shows an unexplained gap during the time of death.",
        "Corridor",
        None,
        "Colonel Slate",
    ),
]


def _resolve_room(house: dict[str, list[str]], preferred_room: str) -> str:
    """Fall back to the Corridor if the preferred room doesn't exist in this house.

    Mirrors ``CharClass._starting_location`` so clue placement degrades the
    same way character placement does when a randomly generated house is
    missing an optional room.
    """
    if preferred_room in house:
        return preferred_room
    return "Corridor"


def build_clues(house: dict[str, list[str]]) -> dict[str, dict]:
    """Build the clue registry for a generated house.

    Args:
        house: The dict returned by ``House_Generator.generate_house``.

    Returns:
        clue_id -> clue dict (see module docstring for the shape).
    """
    clues: dict[str, dict] = {}
    for clue_id, description, preferred_room, corroborates, contradicts in CLUE_DEFINITIONS:
        clues[clue_id] = {
            "description": description,
            "room": _resolve_room(house, preferred_room),
            "discovered": False,
            "corroborates": corroborates,
            "contradicts": contradicts,
        }
    return clues


def discover(clues: dict[str, dict], clue_id: str) -> dict:
    """Mark a clue as found and return it.

    Raises:
        KeyError: if ``clue_id`` isn't a known clue.
    """
    clue = clues[clue_id]
    clue["discovered"] = True
    return clue


def clues_in_room(
    clues: dict[str, dict], room: str, discovered_only: bool = False
) -> list[tuple[str, dict]]:
    """List (clue_id, clue) pairs located in ``room``.

    Use ``discovered_only=False`` (the default) to drive a "search this
    room" mechanic -- you can show the player what's findable without
    revealing the description until they actually search. Use
    ``discovered_only=True`` to list what's already been found there.
    """
    results = []
    for clue_id, clue in clues.items():
        if clue["room"] != room:
            continue
        if discovered_only and not clue["discovered"]:
            continue
        results.append((clue_id, clue))
    return results


def clues_about(clues: dict[str, dict], character_name: str) -> list[tuple[str, dict]]:
    """List every discovered clue that corroborates or contradicts a character.

    Only returns discovered clues -- undiscovered evidence shouldn't be
    usable yet, even by code that already knows it exists.
    """
    results = []
    for clue_id, clue in clues.items():
        if not clue["discovered"]:
            continue
        if clue["corroborates"] == character_name or clue["contradicts"] == character_name:
            results.append((clue_id, clue))
    return results


def undiscovered_count(clues: dict[str, dict]) -> int:
    """How many clues are still hidden -- handy for a progress readout."""
    return sum(1 for clue in clues.values() if not clue["discovered"])


if __name__ == "__main__":
    from CharClass import build_characters
    from House_Generator import generate_house
    import trust_engine

    house = generate_house(seed="demo")
    characters = build_characters(house)

    # Set up a time-of-death snapshot where Hettie and Colonel Slate were
    # alone together -- an unresolved WEAK pair, same as Marrow+Vance in
    # the trust_engine demo.
    for name in characters:
        characters[name][1] = "Mudroom" if name in ("Hettie", "Colonel Slate") else "Library"

    snapshot = trust_engine.snapshot_from_locations(characters)
    trust_engine.update_trust(characters, snapshot, note="time of death")

    print("Unresolved pairs before any clues are found:")
    for a, b in trust_engine.unresolved_pairs(characters):
        print(f"  {a} <-> {b}")

    clues = build_clues(house)
    print(f"\n{undiscovered_count(clues)} clues hidden around the house.")

    # Player searches the Mudroom and finds the footprints clue.
    found = clues_in_room(clues, "Mudroom")
    print(f"\nSearching the Mudroom, found: {[cid for cid, _ in found]}")
    clue = discover(clues, "muddy_footprints")
    print(f"CLUE: {clue['description']}")

    # That clue contradicts Hettie -- use it to resolve her WEAK pair.
    if clue["contradicts"] == "Hettie":
        trust_engine.resolve_weak_pair(
            characters, "Hettie", "Colonel Slate",
            outcome=trust_engine.CONTRADICTED,
            reason=clue["description"],
        )

    print("\nHettie <-> Colonel Slate after the clue:",
          characters["Hettie"][2]["Colonel Slate"])

    print("\nRemaining unresolved pairs:")
    remaining = trust_engine.unresolved_pairs(characters)
    print("  None" if not remaining else "\n".join(f"  {a} <-> {b}" for a, b in remaining))
