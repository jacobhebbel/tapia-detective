"""OpenRouter client used to generate in-character dialogue for agents.

Characters come from ``CharClass.build_characters``:

    character_name -> [role, current_location, trusts, personality]

This module turns that role/personality data plus a conversation history
into a chat-completion request against OpenRouter and returns the
character's next line of dialogue.
"""

from __future__ import annotations

import os

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"


def _load_dotenv(path: str = ".env") -> None:
    """Populate os.environ from a simple KEY=VALUE .env file, if present.

    Existing environment variables always win, so real env vars can still
    override the .env file (e.g. in CI or a shell export).
    """
    if not os.path.exists(path):
        return

    with open(path, encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter can't be reached or returns an unusable reply."""


def _headers() -> dict[str, str]:
    if not OPENROUTER_API_KEY:
        raise OpenRouterError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file "
            "(see .env in the project root)."
        )
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/jacobhebbel/tapia-detective",
        "X-Title": "Tapia Detective",
    }


def build_character_system_prompt(name: str, role: str, personality: str) -> str:
    """Build the system prompt that keeps an agent in character."""
    return (
        f"You are {name}, {role}, a character in a murder-mystery detective "
        f"game. Personality: {personality} "
        "Stay fully in character and speak only as this character would. "
        "Respond with one or two short sentences of spoken dialogue -- no "
        "narration, stage directions, or quotation marks. Never mention that "
        "you are an AI or break character."
    )


def generate_dialogue(
    name: str,
    role: str,
    personality: str,
    conversation: list[dict[str, str]],
    model: str = OPENROUTER_MODEL,
    temperature: float = 0.9,
) -> str:
    """Generate one line of in-character dialogue via OpenRouter.

    Args:
        name: Character name, e.g. from a CharClass character dict key.
        role: The character's role, e.g. ``characters[name][0]``.
        personality: The character's personality, e.g. ``characters[name][3]``.
        conversation: Prior turns as a list of
            ``{"role": "user" | "assistant", "content": str}`` dicts, where
            "user" is the detective/player and "assistant" is this character.
        model: OpenRouter model slug to use.
        temperature: Sampling temperature passed to the model.

    Returns:
        The character's next line of dialogue as plain text.
    """
    messages = [
        {"role": "system", "content": build_character_system_prompt(name, role, personality)},
        *conversation,
    ]

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=_headers(),
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        raise OpenRouterError(f"Could not reach OpenRouter: {exc}") from exc

    if response.status_code != 200:
        raise OpenRouterError(
            f"OpenRouter request failed ({response.status_code}): {response.text}"
        )

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise OpenRouterError(f"Unexpected OpenRouter response: {data}") from exc


if __name__ == "__main__":
    reply = generate_dialogue(
        name="Dr. Teddy Marrow",
        role="Vance's personal physician",
        personality="Calm, clinical, and slightly too helpful; quietly panics "
        "when faced with information he cannot safely ask about.",
        conversation=[{"role": "user", "content": "Where were you at the time of death?"}],
    )
    print(reply)
