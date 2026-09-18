"""Agent and User classes for the top-level Tapia Detective game loop.

``Agent`` represents an NPC suspect wandering the house; ``User`` represents
the human player (the detective). Both classes are deliberately simple --
plain state containers plus the couple of behaviors the game loop needs --
so they stay easy to drive from ``main.py`` without any LLM involvement
(that richer version lives in ``backend/character_agent.py``, which this
module does not touch or depend on).
"""

from __future__ import annotations

import random

import CharClass
import llm_client

# How many past exchanges to replay to the LLM as conversation history.
MAX_HISTORY_TURNS = 12


class Agent:
    """A single NPC suspect: their persona, memory, and location."""

    def __init__(
        self,
        name: str,
        character_prompt: str,
        start_room: str,
        is_killer: bool = False,
    ) -> None:
        self.name = name
        self.character_prompt = character_prompt
        self.current_room = start_room
        self.is_killer = is_killer
        self.conversation_history: list[dict[str, str]] = []

    def remember(self, speaker: str, message: str) -> None:
        """Append one exchange (from ``speaker``) to this agent's history."""
        self.conversation_history.append({"speaker": speaker, "message": message})

    def respond(self, speaker: str, utterance: str) -> str:
        """Ask OpenRouter for this agent's in-character reply and remember it.

        Raises:
            llm_client.MissingAPIKeyError: if OPENROUTER_API_KEY isn't set.
        """
        messages = [{"role": "system", "content": self.character_prompt}]
        for turn in self.conversation_history[-MAX_HISTORY_TURNS:]:
            role = "assistant" if turn["speaker"] == self.name else "user"
            messages.append({"role": role, "content": turn["message"]})
        messages.append({"role": "user", "content": f"{speaker}: {utterance}"})

        reply = llm_client.chat(messages).strip()

        self.remember(speaker, utterance)
        self.remember(self.name, reply)
        return reply

    def move(self, house: dict[str, list[str]]) -> str:
        """Randomly move to a room adjacent to the current one.

        Returns the room the agent ends up in. If the current room has no
        exits, the agent stays put.
        """
        exits = house.get(self.current_room, [])
        if exits:
            self.current_room = random.choice(exits)
        return self.current_room

    def __repr__(self) -> str:
        return f"Agent({self.name!r}, room={self.current_room!r})"


class User:
    """The human player."""

    def __init__(self, name: str = "Detective", start_room: str = "Corridor") -> None:
        self.name = name
        self.current_room = start_room
        self.visited_rooms: set[str] = {start_room}
        self.conversation_history: list[dict[str, str]] = []

    def remember(self, speaker: str, message: str) -> None:
        """Append one exchange (from ``speaker``) to the player's history."""
        self.conversation_history.append({"speaker": speaker, "message": message})

    def move_to(self, room: str) -> None:
        self.current_room = room
        self.visited_rooms.add(room)

    def __repr__(self) -> str:
        return f"User({self.name!r}, room={self.current_room!r})"


def _victim_name() -> str | None:
    for name, role, *_rest in CharClass.CHARACTER_DEFINITIONS:
        if "victim" in role.lower():
            return name
    return None


def _witnesses_by_name(living: list[tuple]) -> dict[str, list[str]]:
    """Map each living character's name to who else shared their room.

    ``living`` is a list of ``CharClass.CHARACTER_DEFINITIONS``-shaped tuples
    with the victim already excluded. Two (or more) people in the same
    time-of-death room can vouch for each other; someone alone in their room
    has no one to vouch for them -- including the killer, who must always be
    placed alone (or alone with the victim) so room logic alone never singles
    them out.
    """
    rooms_by_name = {name: room for name, _role, room, *_rest in living}
    return {
        name: [other for other, other_room in rooms_by_name.items() if other != name and other_room == room]
        for name, room in rooms_by_name.items()
    }


def build_agents(house: dict[str, list[str]]) -> list[Agent]:
    """Build one ``Agent`` per living (non-victim) ``CharClass`` character.

    Also works out, from time-of-death room co-occupancy, who has a witness
    (cleared) and who was alone (unverifiable -- room logic can't rule them
    out). That status is baked into each agent's ``character_prompt`` so the
    LLM roleplays it consistently: witnessed suspects say so plainly, the
    killer lies about an alibi they don't have, and innocent-but-alone
    suspects can offer real, checkable evidence if pressed -- which is what
    should let a careful player tell them apart from the killer.
    """
    victim = _victim_name()
    living = [entry for entry in CharClass.CHARACTER_DEFINITIONS if entry[0] != victim]
    witnesses_by_name = _witnesses_by_name(living)

    agents: list[Agent] = []
    for name, role, preferred_room, personality, alibi_evidence in living:
        start_room = preferred_room if preferred_room in house else "Corridor"
        is_killer = "killer" in role.lower()
        witnesses = witnesses_by_name[name]

        if witnesses:
            alibi = (
                "ALIBI: You were with " + " and ".join(witnesses) + f" in the "
                f"{start_room} at the time of death, and you'll say so plainly "
                "and confidently if asked -- they can back you up."
            )
        elif is_killer:
            alibi = (
                "ALIBI: No one can vouch for you -- you were alone with "
                f"{victim} when you killed {'him' if victim else 'them'}. "
                f"If pressed about your whereabouts, {alibi_evidence}. Never "
                "admit the truth, no matter how directly you're accused."
            )
        else:
            alibi = (
                "ALIBI: No one can vouch for you -- you were alone at the "
                f"time of death. You did nothing wrong, so if someone presses "
                f"you specifically about where you were, {alibi_evidence}. "
                "You don't volunteer this unprompted, though."
            )

        secrecy = (
            "You secretly committed the murder, but you must never admit "
            "this directly or confess outright, no matter how you're "
            "pressed -- deflect, lie convincingly, and stay in character."
            if is_killer
            else "You did not commit the murder, though you may have your "
            "own secrets you'd rather not share."
        )
        character_prompt = (
            f"You are {name}, {role}. {personality} {secrecy} {alibi} "
            "You are being questioned by a detective investigating a murder. "
            "Stay fully in character, speak in the first person, and answer "
            "as a real person would in conversation -- no narration, no "
            "stage directions, no breaking character."
        )
        agents.append(
            Agent(
                name=name,
                character_prompt=character_prompt,
                start_room=start_room,
                is_killer=is_killer,
            )
        )
    return agents
