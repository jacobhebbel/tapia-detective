"""Generate a random house layout as a room-to-neighbors dictionary.

Each dictionary key is a room name, and its value is a list of directly
connected room names. Connections are two-way: if Room A connects to Room B,
Room B also connects to Room A.
"""

from __future__ import annotations

import random
from itertools import combinations
from typing import Iterable


# Rooms that must appear in every generated house.
REQUIRED_ROOMS = [
    "Study",
    "Living Room",
    "Kitchen",
    "Dining Room",
    "Bathroom",
    "Pantry",
    "Laundry Room",
    "Corridor",
]

# The generator randomly chooses a bedroom count within this inclusive range.
BEDROOM_COUNT_RANGE = (2, 4)

# Optional rooms make the building larger. Add a new room here to make it
# eligible for random selection; no other code changes are required.
EXTRA_ROOM_OPTIONS = [
    "Guest Bedroom",
    "Family Room",
    "Home Office",
    "Game Room",
    "Storage Room",
    "Mudroom",
    "Sunroom",
    "Library",
    "Cloakroom",
    "Exercise Room",
]

# Randomly include this many distinct optional rooms.
EXTRA_ROOM_COUNT_RANGE = (3, 6)

# The corridor is the main hub. Every other room connects to it, guaranteeing
# that the whole building is reachable.
HUB_ROOM = "Corridor"

# These plausible direct links are always included in addition to the corridor.
GUARANTEED_CONNECTIONS = [
    ("Living Room", "Dining Room"),
    ("Kitchen", "Dining Room"),
    ("Kitchen", "Pantry"),
]

# Chance of adding another plausible direct connection between two non-hub
# rooms. Increase for a more interconnected building or decrease for a simpler
# corridor-centered layout.
EXTRA_CONNECTION_CHANCE = 0.12


def _connect(
    connections: dict[str, list[str]], room_a: str, room_b: str
) -> None:
    """Add a two-way connection without creating duplicates."""
    if room_b not in connections[room_a]:
        connections[room_a].append(room_b)
    if room_a not in connections[room_b]:
        connections[room_b].append(room_a)


def _choose_extra_rooms(
    rng: random.Random, room_names: Iterable[str]
) -> list[str]:
    """Choose a random number of distinct optional rooms."""
    maximum = min(len(room_names), EXTRA_ROOM_COUNT_RANGE[1])
    count = rng.randint(min(EXTRA_ROOM_COUNT_RANGE), maximum)
    return rng.sample(list(room_names), count)


def generate_house(seed: int | str | None = None) -> dict[str, list[str]]:
    """Return a randomly generated building layout.

    Args:
        seed: Optional seed for repeatable output. When omitted, a new random
            layout is generated on each call.

    Returns:
        A dictionary mapping each room to a list of connected room names.
    """
    rng = random.Random(seed)

    bedroom_count = rng.randint(*BEDROOM_COUNT_RANGE)
    bedrooms = [f"Bedroom {number}" for number in range(1, bedroom_count + 1)]
    extra_rooms = _choose_extra_rooms(rng, EXTRA_ROOM_OPTIONS)

    # Change this list to alter the base set of rooms. Bedrooms and optional
    # rooms are appended automatically.
    room_names = REQUIRED_ROOMS + bedrooms + extra_rooms

    # A list keeps the exact insertion order while the set detects duplicates.
    unique_room_names: list[str] = []
    seen_rooms: set[str] = set()
    for room in room_names:
        if room not in seen_rooms:
            unique_room_names.append(room)
            seen_rooms.add(room)

    connections = {room: [] for room in unique_room_names}

    # Connect every room to the corridor so the graph is fully connected.
    for room in unique_room_names:
        if room != HUB_ROOM:
            _connect(connections, room, HUB_ROOM)

    # Add fixed, realistic connections.
    for room_a, room_b in GUARANTEED_CONNECTIONS:
        _connect(connections, room_a, room_b)

    # Add random secondary routes, but never connect a room to itself.
    non_hub_rooms = [room for room in unique_room_names if room != HUB_ROOM]
    for room_a, room_b in combinations(non_hub_rooms, 2):
        if rng.random() < EXTRA_CONNECTION_CHANCE:
            _connect(connections, room_a, room_b)

    # Sorting makes the generated dictionary easy to read and compare.
    return {
        room: sorted(connected_rooms)
        for room, connected_rooms in connections.items()
    }


def _ascii_box(label: str, width: int) -> list[str]:
    """Create a three-line ASCII box for a room label."""
    border = "+" + "-" * (width + 2) + "+"
    centered_label = label.center(width)
    return [border, f"| {centered_label} |", border]


def print_ascii_map(house: dict[str, list[str]]) -> None:
    """Print a labeled schematic map of the generated house.

    The corridor is shown as a central spine with rooms branching from it.
    Any extra room-to-room connections are listed below the map so the full
    connection graph remains visible.
    """
    if not house:
        print("\nLabeled ASCII map: the house has no rooms.")
        return

    hub = HUB_ROOM if HUB_ROOM in house else next(iter(house))
    branch_rooms = [room for room in house if room != hub]
    label_width = max(len(room) for room in house)

    print("\nLabeled ASCII map")
    print(f"(All rooms connect to {hub}; additional links are listed below.)")

    # Pair rooms on either side of the corridor to keep the map compact.
    for index in range(0, len(branch_rooms), 2):
        left_room = branch_rooms[index]
        right_room = branch_rooms[index + 1] if index + 1 < len(branch_rooms) else None

        left_box = _ascii_box(left_room, label_width) if left_room else None
        right_box = _ascii_box(right_room, label_width) if right_room else None
        hub_box = _ascii_box(hub, label_width)

        left_offset = label_width + 8  # Box width plus the " -- " connector.
        for line_index in range(3):
            left_part = (
                f"{left_box[line_index]} -- "
                if left_box
                else " " * left_offset
            )
            right_part = (
                f" -- {right_box[line_index]}"
                if right_box
                else ""
            )
            print(left_part + hub_box[line_index] + right_part)

        # Continue the corridor spine before the next pair of rooms.
        if index + 2 < len(branch_rooms):
            print(" " * (left_offset + (label_width + 4) // 2) + "|")

    # Show non-corridor links that cannot be represented by the hub-and-spoke
    # schematic above.
    additional_connections: set[tuple[str, str]] = set()
    for room, connected_rooms in house.items():
        for connected_room in connected_rooms:
            if room == hub or connected_room == hub:
                continue
            edge = tuple(sorted((room, connected_room)))
            additional_connections.add(edge)

    print("\nAdditional direct connections:")
    if additional_connections:
        for room_a, room_b in sorted(additional_connections):
            print(f"  {room_a} <--> {room_b}")
    else:
        print("  None")


if __name__ == "__main__":
    generated_house = generate_house()

    print("Generated house connections:")
    for room, connected_rooms in generated_house.items():
        print(f"{room}: {connected_rooms}")

    print_ascii_map(generated_house)
