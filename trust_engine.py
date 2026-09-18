"""Update character ``trusts`` dictionaries from room co-occupancy.

Works directly with the structures already produced by ``CharClass`` and
``House_Generator``:

    characters: character_name -> [role, current_location, trusts, personality]
    house:      room_name -> list of connected room names

This module adds no new data structures of its own beyond what ``trusts``
already is. It just decides, given a snapshot of who is in which room during
a given time window, how each character's ``trusts`` dictionary should be
updated.

THE CORE RULE (this is the answer to "how do you tell an innocent pair from
an innocent+killer pair who were alone together"): co-occupancy alone can
NEVER produce that answer, so this module refuses to try. It recognizes
exactly three outcomes from room occupancy alone:

    3+ people in a room together  -> mutually well-alibied ("strong")
    2 people in a room together   -> mutually vouching, but UNRESOLVED
                                      ("weak") -- neither cleared nor accused
    1 person alone in a room      -> no alibi at all ("none")

A "weak" trust entry is a flag meaning "needs outside evidence," not a
verdict. Nothing in this module ever upgrades a "weak" pair to "strong" --
that upgrade may only come from something outside co-occupancy (a witness
in another room, a physical clue, a contradiction in a later statement,
etc.), which is why ``resolve_weak_pair`` below takes an explicit reason
instead of inferring one.
"""

from __future__ import annotations

# Trust status values stored in each character's ``trusts[other_name]`` entry.
STRONG = "strong"        # corroborated by a third party (group of 3+)
WEAK = "weak"             # only the two of them -- unresolved, not exculpatory
NONE_ALIBI = "none"      # alone, no alibi to check
CONTRADICTED = "contradicted"  # an outside source disproved the claim
CLEARED = "cleared"      # an outside source confirmed the claim


def snapshot_from_locations(characters: dict[str, list]) -> dict[str, list[str]]:
    """Group character names by their current ``current_location``.

    Convenience helper for the common case where you just want to use each
    character's live position (as tracked in ``characters[name][1]``) as the
    occupancy snapshot for this time step, rather than building one by hand.

    Returns:
        room_name -> list of character names currently in that room.
    """
    rooms: dict[str, list[str]] = {}
    for name, details in characters.items():
        room = details[1]
        rooms.setdefault(room, []).append(name)
    return rooms


def update_trust(
    characters: dict[str, list],
    room_snapshot: dict[str, list[str]],
    note: str = "",
) -> dict[str, str]:
    """Update ``trusts`` for every character based on one occupancy snapshot.

    Args:
        characters: The dict produced by ``CharClass.build_characters``.
            Modified in place: each occupant's ``trusts`` dict (index 2)
            gets an entry for every other occupant of their room.
        room_snapshot: room_name -> list of character names present, e.g.
            the output of ``snapshot_from_locations`` or a hand-built dict
            for a specific time-of-death window.
        note: Optional label (e.g. "20:00-20:30, time of death") stored
            alongside the status so you can see *when* a trust entry was
            set, in case it needs to be re-examined later.

    Returns:
        A summary dict of character_name -> status assigned this round,
        for callers who want to print/log what just happened without
        digging through every ``trusts`` dict.
    """
    summary: dict[str, str] = {}

    for room, occupants in room_snapshot.items():
        if len(occupants) >= 3:
            status = STRONG
        elif len(occupants) == 2:
            status = WEAK
        else:
            # 0 or 1 occupants: nothing to record between two people.
            for solo in occupants:
                summary[solo] = NONE_ALIBI
            continue

        for name in occupants:
            trusts = characters[name][2]
            for other in occupants:
                if other == name:
                    continue
                trusts[other] = {"status": status, "room": room, "note": note}
            summary[name] = status

    return summary


def resolve_weak_pair(
    characters: dict[str, list],
    char_a: str,
    char_b: str,
    outcome: str,
    reason: str,
) -> None:
    """Resolve a WEAK (two-person, unverifiable) trust entry using outside evidence.

    This is the only function allowed to move a pair out of WEAK, and it
    requires the caller to supply a concrete, non-testimonial reason (a
    clue, a third-party sighting, a contradicted statement, etc.) rather
    than inferring one -- so it's impossible to accidentally "trust someone
    into innocence."

    Args:
        characters: The characters dict, modified in place.
        char_a, char_b: The two characters whose mutual trust entry is
            being resolved.
        outcome: Either ``CLEARED`` or ``CONTRADICTED``.
        reason: A short human-readable justification, stored on both
            sides' trust entries (e.g. "muddy footprints outside the
            Kitchen contradict Hettie's claim").
    """
    if outcome not in (CLEARED, CONTRADICTED):
        raise ValueError("outcome must be trust_engine.CLEARED or trust_engine.CONTRADICTED")

    for name, other in ((char_a, char_b), (char_b, char_a)):
        trusts = characters[name][2]
        existing = trusts.get(other, {})
        if existing.get("status") != WEAK:
            raise ValueError(
                f"{name} and {other} do not currently have a WEAK trust entry "
                f"to resolve (found: {existing.get('status')!r}). Only a WEAK "
                f"pair can be resolved this way."
            )
        trusts[other] = {"status": outcome, "room": existing.get("room"), "note": reason}


def unresolved_pairs(characters: dict[str, list]) -> list[tuple[str, str]]:
    """List every mutual WEAK trust pair still awaiting outside evidence.

    Useful as a detective to-do list: these are exactly the relationships
    that room-occupancy logic could not settle on its own.
    """
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for name, details in characters.items():
        trusts = details[2]
        for other, entry in trusts.items():
            if entry.get("status") == WEAK:
                key = tuple(sorted((name, other)))
                if key not in seen:
                    seen.add(key)
                    pairs.append(key)
    return pairs


def alibi_strength(characters: dict[str, list], name: str) -> str:
    """Return the single weakest (least-verified) status a character currently has.

    A character can appear in several rooms across several snapshots as the
    story unfolds; this reports their most suspicious standing so far, using
    the natural ordering: contradicted > none > weak > strong > cleared.
    """
    order = {CONTRADICTED: 0, NONE_ALIBI: 1, WEAK: 2, STRONG: 3, CLEARED: 4}
    trusts = characters[name][2]
    if not trusts:
        return NONE_ALIBI
    worst = min((entry["status"] for entry in trusts.values()), key=lambda s: order[s])
    return worst


if __name__ == "__main__":
    from CharClass import build_characters
    from House_Generator import generate_house

    house = generate_house(seed="demo")
    characters = build_characters(house)

    # Simulate everyone's location during the time-of-death window by hand,
    # the way a game loop would set current_location before calling this.
    tod_locations = {
        "Ignatius Vance": "Study",
        "Dr. Teddy Marrow": "Study",          # alone with the victim -- WEAK, not proof
        "Wren": "Game Room",
        "Hettie": "Kitchen",
        "Odette": "Library",
        "Barnaby": "Library",                  # alone with Odette -- also WEAK
        "Colonel Slate": "Mudroom",
    }
    for name, room in tod_locations.items():
        characters[name][1] = room

    snapshot = snapshot_from_locations(characters)
    summary = update_trust(characters, snapshot, note="20:00-20:30 time of death")

    print("Trust status after the time-of-death window:")
    for name, status in summary.items():
        print(f"  {name:20s} -> {status}")

    print("\nUnresolved (2-person, unverifiable) pairs needing evidence:")
    for a, b in unresolved_pairs(characters):
        print(f"  {a} <-> {b}")

    # Show that WEAK cannot be talked away -- only resolved with a reason.
    resolve_weak_pair(
        characters, "Odette", "Barnaby",
        outcome=CLEARED,
        reason="Colonel Slate independently confirms hearing them both in the Library.",
    )
    print("\nAfter outside corroboration, Odette/Barnaby:",
          characters["Odette"][2]["Barnaby"])

    print("\nRemaining unresolved pairs:")
    for a, b in unresolved_pairs(characters):
        print(f"  {a} <-> {b}")
