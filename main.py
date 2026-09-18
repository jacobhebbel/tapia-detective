"""Text-input game loop for the Tapia Detective game."""

from __future__ import annotations

import llm_client
from agents import Agent, User, build_agents
from House_Generator import generate_house, print_ascii_map

TITLE_ART = r"""
 _____  _    ____ ___    _
|_   _|/ \  |  _ \_ _|  / \
  | | / _ \ | |_) | |  / _ \
  | |/ ___ \|  __/| | / ___ \
  |_/_/   \_\_|  |___/_/   \_\

 ____  _____ _____ _____ ____ _____ _____     _______
|  _ \| ____|_   _| ____/ ___|_   _|_ _\ \   / / ____|
| | | |  _|   | | |  _|| |     | |  | | \ \ / /|  _|
| |_| | |___  | | | |__| |___  | |  | |  \ V / | |___
|____/|_____| |_| |_____\____| |_| |___|  \_/  |_____|
"""

HOW_TO_PLAY = """\
A murder has taken place in this house, and you're the detective on the
scene. Explore the rooms, question the people you find, and piece together
what really happened -- then make your one accusation. Choose carefully:
you only get one shot.

How it works:
  /move <room>     - walk to a connected room
  /look            - see your surroundings and who else is here
  /talkto <name>   - start a conversation with someone in your room
  /respond <text>  - say something once you're in a conversation
  /leave           - step away from a conversation
  /accuse <name>   - make your one and only accusation

Suspects go about their business between conversations, but the world
only moves forward when you actually talk to someone -- so take your
time exploring.

Type /help any time for the full command list.
"""


def show_title_screen() -> None:
    print(TITLE_ART)
    print(HOW_TO_PLAY)


class GameState:

    def __init__(self, house: dict[str, list[str]], start_room: str = "Corridor") -> None:
        self.house = house
        self.user = User(start_room=start_room)
        self.agents: list[Agent] = build_agents(house)
        self.active_conversation: Agent | None = None
        self.turn_count = 0
        self.running = True

    @property
    def current_room(self) -> str:
        return self.user.current_room

    def move_to(self, room: str) -> None:
        self.user.move_to(room)
        self.turn_count += 1

    def exits(self) -> list[str]:
        return self.house.get(self.current_room, [])

    def agents_in_room(self, room: str | None = None) -> list[Agent]:
        room = self.current_room if room is None else room
        return [agent for agent in self.agents if agent.current_room == room]

    def find_agent(self, name: str) -> Agent | None:
        lowered = name.lower()
        for agent in self.agents:
            if agent.name.lower() == lowered:
                return agent
        return None

    def advance_agents(self) -> None:
        for agent in self.agents:
            agent.move(self.house)


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
    print("  /talkto <agentName>                   - talk to someone")
    print("  /respond <text>                       - respond in a conversation")
    print("  /leave                                - leave a conversation")
    print("  /accuse <suspect>                     - make an accusation")
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

    others = state.agents_in_room()
    if others:
        print("Also here: " + ", ".join(agent.name for agent in others))


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

    state.active_conversation = None
    state.move_to(target_room)
    print(f"You move to the {target_room}.")


def handle_talkto(state: GameState, args: list[str]) -> None:
    if not args:
        print("Talk to whom? Usage: /talkto <agentName>")
        return

    target_name = " ".join(args)
    agent = state.find_agent(target_name)

    if agent is None:
        print(f"There's no one named {target_name!r} in this story.")
        return

    if agent.current_room != state.current_room:
        print(f"{agent.name} isn't here right now.")
        return

    state.active_conversation = agent
    print(f"{state.user.name} approaches {agent.name}.")
    print(f"You are now talking to {agent.name}. Use /respond <text> to speak, /leave to stop.")

    # The world only moves forward once you actually engage someone in
    # conversation -- walking around and looking never scatters the cast,
    # so you don't end up chasing people from room to room.
    state.advance_agents()


def handle_respond(state: GameState, args: list[str]) -> None:
    if state.active_conversation is None:
        print("You're not talking to anyone. Use /talkto <agentName> first.")
        return

    if not args:
        print("Respond with what? Usage: /respond <text>")
        return

    text = " ".join(args)
    agent = state.active_conversation
    print(f'You say to {agent.name}: "{text}"')

    try:
        reply = agent.respond(state.user.name, text)
    except llm_client.MissingAPIKeyError:
        reply = (
            f"({agent.name} eyes you cautiously but says nothing -- no "
            "OPENROUTER_API_KEY is configured, so live dialogue is off.)"
        )
        agent.remember(state.user.name, text)
        agent.remember(agent.name, reply)
    except Exception as exc:  # network/API errors shouldn't crash the game
        reply = f"({agent.name} hesitates -- something went wrong reaching them: {exc})"
        agent.remember(state.user.name, text)
        agent.remember(agent.name, reply)

    state.user.remember(state.user.name, text)
    state.user.remember(agent.name, reply)
    print(f"{agent.name}: {reply}")


def handle_leave(state: GameState) -> None:
    if state.active_conversation is None:
        print("You're not in a conversation.")
        return

    print(f"You step away from {state.active_conversation.name}.")
    state.active_conversation = None


def handle_accuse(state: GameState, args: list[str]) -> None:
    if not args:
        print("Accuse whom? Usage: /accuse <suspect>")
        return

    target_name = " ".join(args)
    agent = state.find_agent(target_name)

    if agent is None:
        print(f"There's no one named {target_name!r} in this story.")
        return

    if agent.is_killer:
        print(f"You accuse {agent.name}... and you're right! Case closed.")
    else:
        print(f"You accuse {agent.name}... but you're wrong. The real killer walks free.")

    state.running = False


def handle_quit(state: GameState) -> None:
    state.running = False


def main() -> None:
    house = generate_house()
    state = GameState(house, start_room="Corridor")
    show_title_screen()

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
            handle_talkto(state, args)
        elif command == "respond":
            handle_respond(state, args)
        elif command == "leave":
            handle_leave(state)
        elif command == "accuse":
            handle_accuse(state, args)
        else:
            print(f"Unknown command: {parts[0]!r}. Type /help for a list of commands.")

    print("Thanks for playing.")


if __name__ == "__main__":
    main()
