"""Generate a static house layout as a room-to-neighbors dictionary.

Each dictionary key is a room name, and its value is a list of directly
connected room names. Connections are two-way: if Room A connects to Room B,
Room B also connects to Room A.
"""

from __future__ import annotations

# Static house definition
STATIC_HOUSE = {
    "Living Room": ["Kitchen", "Bedroom"],
    "Kitchen": ["Living Room"],
    "Bedroom": ["Living Room", "Bathroom"],
    "Bathroom": ["Bedroom"]
}


def generate_house(seed: int | str | None = None) -> dict[str, list[str]]:
    """Return the static house layout.

    Args:
        seed: Ignored, kept for compatibility.

    Returns:
        A dictionary mapping each room to a list of connected room names.
    """
    return STATIC_HOUSE


def _wrap_connections(connections: list[str], width: int) -> list[str]:
    """Wrap a room's connection names so they fit inside an ASCII box."""
    if not connections:
        return ["doors: none"]

    lines: list[str] = []
    current_line = "doors:"
    for connection in connections:
        separator = " " if current_line == "doors:" else ", "
        candidate = current_line + separator + connection

        # Room names are no wider than the box, so a single name always fits.
        if current_line != "doors:" and len(candidate) > width:
            lines.append(current_line)
            current_line = connection
        else:
            current_line = candidate

    lines.append(current_line)
    return lines


def _ascii_room_card(
    label: str, width: int, connections: list[str]
) -> list[str]:
    """Create one labeled room card with its complete door list."""
    border = "+" + "-" * (width + 2) + "+"
    lines = [border, f"| {label.ljust(width)} |"]
    lines.extend(
        f"| {connection_line.ljust(width)} |"
        for connection_line in _wrap_connections(connections, width)
    )
    lines.append(border)
    return lines


def _pad_card(card: list[str], width: int, target_height: int) -> list[str]:
    """Pad a room card so paired cards stay aligned."""
    padded = list(card)
    while len(padded) < target_height:
        padded.insert(-1, f"| {'':<{width}} |")
    return padded


def print_ascii_map(house: dict[str, list[str]]) -> None:
    """Print a labeled ASCII map containing the complete connection graph.

    Each room card lists every directly connected room. This keeps the map
    self-contained and avoids a separate list of additional connections.
    """
    if not house:
        print("\nLabeled ASCII map: the house has no rooms.")
        return

    # Use first room as hub for ordering
    hub = next(iter(house))
    rooms = [hub] + [room for room in house if room != hub]

    longest_connection = max(
        (len(connection) for neighbors in house.values() for connection in neighbors),
        default=0,
    )
    card_width = max(
        len("doors:"),
        max(len(room) for room in house),
        longest_connection,
    ) + 2

    cards = {
        room: _ascii_room_card(room, card_width, house[room])
        for room in rooms
    }
    target_height = max(len(card) for card in cards.values())
    padded_cards = {
        room: _pad_card(card, card_width, target_height)
        for room, card in cards.items()
    }

    print("\nLabeled ASCII map")
    print('(Each box is a room; "doors" lists every directly connected room.)')

    # Display two room cards per row.
    for index in range(0, len(rooms), 2):
        left_room = rooms[index]
        right_room = rooms[index + 1] if index + 1 < len(rooms) else None
        left_card = padded_cards[left_room]
        right_card = padded_cards[right_room] if right_room else []

        for line_index in range(target_height):
            left_line = left_card[line_index]
            right_line = right_card[line_index] if right_card else ""
            print(left_line + ("    " if right_card else "") + right_line)


if __name__ == "__main__":
    generated_house = generate_house()

    print("Generated house connections:")
    for room, connected_rooms in generated_house.items():
        print(f"{room}: {connected_rooms}")

    print_ascii_map(generated_house)