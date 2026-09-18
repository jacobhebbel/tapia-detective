"""Text-input game loop for the Tapia Detective game."""

from __future__ import annotations

from House_Generator import generate_house, print_ascii_map


class GameState:

    def __init__(self, house: dict[str, list[str]], start_room: str = "Corridor") -> None:
        self.house = house
        self.current_room = start_room
        self.visited_rooms: set[str] = {start_room}
        self.turn_count = 0
        self.running = True

    def move_to(self, room: str) -> None:
        self.current_room = room
        self.visited_rooms.add(room)
        self.turn_count += 1

    def exits(self) -> list[str]:
        return self.house.get(self.current_room, [])


def normalize_command(token: str) -> str:
    return token.lstrip("/").lower()


def _resolve_room_name(house: dict[str, list[str]], name: str) -> str | None:
    lowered = name.lower()
    for room in house:
        if room.lower() == lowered:
            return room
    return None


def handle_help() -> None:
    print("Available commands:")
    print("  /look                                - describe your current room and exits")
    print("  /map                                  - show the house map")
    print("  /move <room>                          - move to a connected room")
    print("  /talkto <agentName>                   - talk to someone (not yet implemented)")
    print("  /respond <text>                       - respond in a conversation (not yet implemented)")
    print("  /leave                                - leave a conversation (not yet implemented)")
    print("  /accuse <suspect>                     - make an accusation (not yet implemented)")
    print("  /help                                 - show this help text")
    print("  /quit, /exit                          - quit the game")


def handle_map(state: GameState) -> None:
    print_ascii_map(state.house)


def handle_look(state: GameState) -> None:
    exits = state.exits()
    print(f"You are in the {state.current_room}.")
    if exits:
        print("Exits: " + ", ".join(exits))
    else:
        print("There are no visible exits.")


def handle_move(state: GameState, args: list[str]) -> None:
    if not args:
        print("Move where? Usage: /move <room name>")
        return

    target_input = " ".join(args)
    target_room = _resolve_room_name(state.house, target_input)

    if target_room is None:
        print(f"Unknown room: {target_input!r}.")
        return

    if target_room not in state.exits():
        print(f"You can't get to {target_room} from {state.current_room}.")
        print(f"From here you can reach: {', '.join(state.exits())}")
        return

    state.move_to(target_room)
    print(f"You move to the {target_room}.")


def handle_talkto(args: list[str]) -> None:
    print("Talking to agents is not yet implemented.")


def handle_respond(args: list[str]) -> None:
    print("Responding is not yet implemented.")


def handle_leave() -> None:
    print("Leaving a conversation is not yet implemented.")


def handle_accuse(args: list[str]) -> None:
    print("Making an accusation is not yet implemented.")


def handle_quit(state: GameState) -> None:
    state.running = False


def main() -> None:
    house = generate_house()
    state = GameState(house, start_room="Corridor")
    print("Welcome to Tapia Detective. Type /help for commands.")

    while state.running:
        raw = input("> ").strip()
        if not raw:
            continue

        parts = raw.split()
        command = normalize_command(parts[0])
        args = parts[1:]

        if command in ("quit", "exit"):
            handle_quit(state)
        elif command == "help":
            handle_help()
        elif command == "map":
            handle_map(state)
        elif command == "look":
            handle_look(state)
        elif command == "move":
            handle_move(state, args)
        elif command == "talkto":
            handle_talkto(args)
        elif command == "respond":
            handle_respond(args)
        elif command == "leave":
            handle_leave()
        elif command == "accuse":
            handle_accuse(args)
        else:
            print(f"Unknown command: {parts[0]!r}. Type /help for a list of commands.")

    print("Thanks for playing.")


if __name__ == "__main__":
    main()
