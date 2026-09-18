# 🔎 Tapia Detective

**A murder-mystery text adventure where every suspect is an AI agent with its own personality, memory, secrets, and alibi.**

*Built for the Tapia Conference 2026 Hackathon.*

---

## Project summary

> **Tapia Detective** is a terminal-based murder mystery powered by a multi-agent LLM system. Six suspects, each run by its own large-language-model agent, move around a house after the death of puzzle magnate Ignatius Vance. You play the detective: explore the rooms, question anyone you find, and make **one** accusation.
>
> No suspect is scripted. Each agent gets a unique persona, private memory of its conversations with you, and an alibi that the game **computes from where everyone was at the time of death**. Innocent suspects who were alone can back up their stories with real, checkable evidence. The killer can't, so they lie, deflect, and get evasive under pressure. To win, you have to catch the inconsistency, the same way a real detective would.

---

## Table of contents

1. [Quick start](#quick-start)
2. [How to play](#how-to-play)
3. [The case & the cast](#the-case--the-cast)
4. [How the solution works](#how-the-solution-works)
5. [Agentic framework, LLMs & tools](#agentic-framework-llms--tools)
6. [Execution environment](#execution-environment)
7. [Evaluation, iterative refinement & context complexity](#evaluation-iterative-refinement--context-complexity)
8. [Repository layout](#repository-layout)
9. [Development journey](#development-journey)
10. [Limitations & future work](#limitations--future-work)
11. [Team](#team)

---

## Quick start

```bash
git clone https://github.com/jacobhebbel/tapia-detective.git
cd tapia-detective
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with your [OpenRouter API key](https://openrouter.ai/keys):

```env
OPENROUTER_API_KEY=sk-or-v1-...
# Optional: choose any OpenRouter model (default: anthropic/claude-3.5-sonnet)
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
```

Then run:

```bash
python main.py
```

> **No key? No problem.** The game still starts and you can explore, open the map, and accuse suspects. Suspects just won't speak: they'll "eye you cautiously" until a key is configured.

---

## How to play

You start in the **Corridor**. Type commands at the `>` prompt:

| Command | What it does |
|---|---|
| `/look` | Describe your room, its exits, and who is here |
| `/map` | Show an ASCII map of the house |
| `/move <room>` | Walk to a connected room |
| `/talkto <name>` | Start a conversation with someone in your room |
| `/respond <text>` | Say something to the person you're talking to |
| `/leave` | Step away from the conversation |
| `/accuse <name>` | Make your **one and only** accusation (ends the game) |
| `/help` | List all commands |
| `/quit` or `/exit` | Quit |

**Example session**

```text
> /look
You are in the Corridor.
Exits: Living Room, Kitchen, Bedroom, Bathroom
Also here: Colonel Slate
> /talkto Colonel Slate
You are now talking to Colonel Slate. Use /respond <text> to speak, /leave to stop.
> /respond Where were you at the time of death?
Colonel Slate: ...
```

**Tip:** Suspects wander between rooms, but the world only moves forward when you **start a conversation**. Walking around and looking never scatters the cast, so you won't end up chasing people from room to room.

---

## The case & the cast

The house has five connected rooms:

```
                 ┌─────────────┐
       ┌─────────┤  Corridor   ├─────────┐
       │         └──┬───────┬──┘         │
 ┌─────┴─────┐      │       │      ┌─────┴─────┐
 │  Kitchen  ├──┐   │       │   ┌──┤ Bathroom  │
 └───────────┘  │   │       │   │  └───────────┘
            ┌───┴───┴──┐ ┌──┴───┴──┐
            │  Living  ├─┤ Bedroom │
            │   Room   │ └─────────┘
            └──────────┘
```

**Ignatius Vance**, a wealthy and secretive puzzle magnate, is dead. Six people were in the house:

| Suspect | Role | Personality |
|---|---|---|
| **Dr. Teddy Marrow** | Vance's personal physician | Calm, clinical, slightly *too* helpful |
| **Wren** | Vance's estranged granddaughter | Broke, defensive, lies about when she arrived |
| **Hettie** | Housekeeper | Loyal and protective; would tamper with a scene for someone she loves |
| **Odette** | Rival game designer | Playful chaos agent who spreads misleading theories |
| **Barnaby** | Vance's old friend | Anxious and evasive; hiding an old financial scandal |
| **Colonel Slate** | Head of island security | Procedural, guarded, a bad liar; protective of her relationship with Hettie |

Every suspect has something to hide, so everyone looks a little guilty. The game is about separating *"hiding a secret"* from *"hiding a murder."*

<details>
<summary><b>⚠️ Spoiler: how the mystery is designed to be solved</b></summary>

- **Wren & Hettie** were together in the Kitchen. They vouch for each other and are cleared.
- **Odette** (Bedroom), **Barnaby** (Bathroom), and **Colonel Slate** (Corridor) were each alone. Each can offer checkable evidence if pressed: timestamped phone messages, a bin the housekeeper emptied, and security-camera footage.
- **Dr. Teddy Marrow** was alone with the victim in the Living Room. **He is the killer.** His cover story claims he was alone in the Bathroom treating a headache. He has no evidence, he gets evasive about details, and Barnaby was the one actually in the Bathroom.

</details>

---

## How the solution works

Tapia Detective is a **multi-agent system**. Each suspect is an independent LLM agent. A Python game engine owns the world state and gives every agent only the context it's allowed to have.

```
                         ┌──────────────────────────────┐
   Player (terminal) ──▶ │   main.py : GameState loop   │
                         │  parse command → dispatch    │
                         └───┬─────────────┬────────────┘
                             │             │
             world state     │             │   /respond
   ┌─────────────────────────┴──┐     ┌────┴──────────────────────────┐
   │ House_Generator.py         │     │ agents.py : Agent (×6)        │
   │  room graph + ASCII map    │     │  • persona system prompt      │
   │ CharClass.py               │     │  • private memory (history)   │
   │  cast, roles, rooms,       │     │  • location + random movement │
   │  personalities, evidence   │     │  • is_killer (engine-only)    │
   └────────────────────────────┘     └────┬──────────────────────────┘
                                           │ chat(messages)
                                      ┌────┴──────────────────────────┐
                                      │ llm_client.py                 │
                                      │  OpenAI SDK → OpenRouter API  │
                                      │  → Claude 3.5 Sonnet (default)│
                                      └───────────────────────────────┘
```

### 1. World model (`House_Generator.py`)
The house is a **graph**: a dictionary that maps each room to its neighboring rooms, with two-way connections. Movement checks for players and agents both use this graph. `print_ascii_map` draws it as labeled room cards.

### 2. Character definitions (`CharClass.py`)
One list, `CHARACTER_DEFINITIONS`, is the single source of truth for the story. It stores each character's **name, role, time-of-death room, personality, and alibi evidence**. To add, remove, or rewrite a suspect, you only edit this list.

### 3. Alibi reasoning: the core mechanic (`agents.build_agents`)
Alibis aren't hand-written. The engine **derives** them from room co-occupancy at the time of death:

| Situation | What the engine tells the agent |
|---|---|
| Shared a room with another living person | "You were with *X*. Say so plainly; they can back you up." |
| Alone and **innocent** | "No one can vouch for you. If pressed, offer this *real, checkable* evidence. Don't volunteer it unprompted." |
| Alone and the **killer** | "You were alone with the victim. If pressed, give this *false* story. Never admit the truth." |

This keeps the puzzle **fair and consistent**. The killer is always placed alone, so room logic alone never gives them away. The only way to crack the case is to interrogate suspects and weigh the quality of their evidence.

### 4. LLM agents (`agents.Agent`)
Each `Agent` builds its own prompt from scratch every turn:

1. A **system prompt**: persona + role + secrecy rules + computed alibi + role-play rules ("first person, no narration, never break character").
2. The agent's **own conversation history**: the last 12 messages, mapped to `user` / `assistant` roles.
3. The detective's **new line**.

The agent's reply is saved to its private memory, so a suspect remembers what you said and what it claimed earlier, even if you leave and come back.

**Information hiding:** The ground truth (`is_killer`) lives in the engine. Agents never see another suspect's prompt or memory. Innocent agents are only told they are innocent, so they cannot leak who the killer is.

### 5. Game loop (`main.py`)
A `GameState` object tracks the house, the player, the six agents, the active conversation, and a turn counter. Commands are parsed and dispatched to handlers. Agents take a random step to an adjacent room each time the player starts a conversation. `/accuse` checks the chosen suspect against ground truth and ends the game.

### 6. Graceful degradation
If the API key is missing, `MissingAPIKeyError` is caught and the suspect replies with an in-world "stays silent" line. Any network or API error is also caught and shown in-character, so the game never crashes mid-interrogation.

---

## Agentic framework, LLMs & tools

| Category | What we used |
|---|---|
| **Agent framework** | **Custom, lightweight multi-agent architecture in plain Python.** We did not use LangChain, CrewAI, AutoGen, or another orchestration framework. Each agent is a Python object with a persona, memory, and location, and the game engine is the orchestrator. We chose this for full control over what each agent sees, so there's no hidden context that could leak the solution. |
| **LLM (default)** | **Anthropic Claude 3.5 Sonnet** (`anthropic/claude-3.5-sonnet`), temperature `0.8`. |
| **LLM gateway** | **OpenRouter**: one API key, any model. You can swap models (e.g. `openai/gpt-4o-mini`, Llama, Mistral) by setting `OPENROUTER_MODEL`, with no code changes. |
| **LLM SDK** | **OpenAI Python SDK** (`openai`), pointed at OpenRouter's OpenAI-compatible endpoint. |
| **Alternate client** | `api.py`: a standalone OpenRouter client using `requests` directly (default `openai/gpt-4o-mini`, temperature `0.9`). This was our first integration. It's kept for reference and isn't used by the main game loop. |
| **Config / secrets** | `python-dotenv` loads `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` from `.env`, which is git-ignored. |
| **Language / stdlib** | Python 3 with type hints; `random` for agent movement. |
| **Collaboration** | Git + GitHub (branches, pull requests, merges). |
| **AI-assisted development** | Parts of the codebase were built with AI coding assistants, including Claude (credited as co-author on the final rebuild commit). |

---

## Execution environment

| Aspect | Details |
|---|---|
| **Interface** | Interactive command-line app (text in, text out); runs in any terminal |
| **Runtime** | Python **3.10+** (tested on Python 3.11) |
| **Operating systems** | macOS, Linux, Windows |
| **Dependencies** | `pip install -r requirements.txt` (key packages: `openai`, `python-dotenv`) |
| **Compute** | Runs locally on a standard laptop. No GPU, no database, no web server. |
| **Model inference** | Remote, in the cloud via the OpenRouter API (requires internet) |
| **Configuration** | Environment variables / `.env`: `OPENROUTER_API_KEY` (required for dialogue), `OPENROUTER_MODEL` (optional), `OPENROUTER_REFERER` and `OPENROUTER_TITLE` (optional request headers) |
| **Offline mode** | Without a key, exploration, map, movement, and accusation still work; dialogue is disabled |
| **Cost / latency** | One LLM call per `/respond`, with small prompts (see below), so each turn is cheap and fast |

---

## Evaluation, iterative refinement & context complexity

### ✅ Evaluation: *Yes (built into the game), with automated testing in an earlier prototype*
- **Ground-truth evaluation in the game:** The final accusation is checked against the engine's hidden ground truth (`is_killer`). The player gets a clear win/lose result, so every playthrough evaluates whether the agents gave the player enough to solve the case.
- **Designed-in fairness check:** The alibi rules guarantee the mystery is solvable. The killer is always alone, and every innocent suspect who was alone has real evidence to offer. This was confirmed by manual playtesting.
- **Earlier prototype:** An earlier web version had an **automated leak-probe test suite**. It replayed adversarial questions against every character, scanned replies for protected "canary" facts with a runtime leak guard, and reported "0 violations across N turns." It was removed in the final rebuild to keep the project small (see [Development journey](#development-journey)).

### 🔁 Iterative refinement: *Yes*
- **In the game:** Each agent's memory grows with every exchange. Its later answers build on what it said earlier and what you've already revealed, and the detective refines their theory turn by turn. Agents do not self-critique or rewrite their own answers; refinement happens through the ongoing conversation.
- **In development:** The project went through several iterations: a prototype game loop and OpenRouter client, then a full web app with trust graphs and disclosure tiers, then a focused terminal rebuild. That rebuild also fixed a bug where most suspects' rooms didn't exist in the map, which collapsed the whole cast into the Corridor and broke the alibi logic.

### 🧠 Context complexity: *Moderate*
- **Multi-agent, with information asymmetry:** 6 separate agents, each with an **isolated** context. No agent can see another's prompt or memory, and only the engine knows the solution.
- **Per-call context:** a system prompt of about **105–160 words** (≈150–250 tokens) + up to **12** prior messages + the new line, so usually **under ~2K tokens** per call.
- **Layered instructions:** Each system prompt combines persona, role, a secrecy rule, a **computed** alibi (one of three types), and role-play rules. The killer's prompt is the most complex: the agent must keep a lie consistent under direct accusation.
- **Dynamic state:** Agent locations, conversation memory, and the active conversation all change during play. The engine turns this into a small, focused prompt for each call.

---

## Repository layout

```
tapia-detective/
├── main.py              # Game loop: title screen, command parser, GameState, handlers
├── agents.py            # Agent (LLM-backed suspect) & User (detective); alibi logic in build_agents()
├── llm_client.py        # OpenRouter chat client via the OpenAI SDK (used by the game)
├── CharClass.py         # Cast definitions: name, role, room, personality, alibi evidence
├── House_Generator.py   # House room graph + ASCII map renderer
├── api.py               # Earlier standalone OpenRouter client (requests-based, not used by main.py)
├── requirements.txt     # Python dependencies
├── LICENSE
└── README.md
```

---

## Development journey

We built all of this during the hackathon, in stages:

1. **Foundations:** A house generator (room graph), character definitions, a trust-engine draft, and a first terminal game loop.
2. **LLM integration:** Added OpenRouter so suspects could talk (`api.py`, merged through our first pull request).
3. **Ambitious prototype:** A FastAPI + WebSocket web app with a browser UI, JSON character dossiers, a trust graph, a tiered "disclosure ladder" for secrets, a clue engine, a runtime leak guard, and an automated leak-test dashboard (about 6,800 lines).
4. **Focus & rebuild:** We stripped the project back to a self-contained terminal game. The core idea stayed the same: independent agents with private knowledge, plus alibis computed from the world state. The result is about 500 lines that are easy to run, read, and demo.

The lesson: **a smaller, correct agent system beat a bigger one we couldn't finish polishing in time.**

---

## Limitations & future work

- **Agent-to-agent interaction:** Suspects currently talk only to the detective. Next step: let them overhear, gossip, and react to each other.
- **Trust & disclosure:** `CharClass` already stores a `trusts` map for each character. Next step: wire it in so pressure and rapport unlock secrets gradually, as in our prototype.
- **Clues & evidence board:** Add physical clues to find in rooms and a notebook for collected evidence.
- **Automated evaluation:** Bring back the leak-probe tests, and add LLM-as-judge checks for character consistency and "the killer never confesses."
- **Replayability:** Randomize the killer, rooms, and alibis for each playthrough (`generate_house` already accepts a `seed`).
- **Richer map:** Add more rooms, and procedurally generated houses.

---

## Team

Built for the **Tapia Conference Hackathon** by:

- [@jacobhebbel](https://github.com/jacobhebbel)
- [@Cashmin](https://github.com/Cashmin)
- [@cenjingwang](https://github.com/cenjingwang)
- [@Ameyabarve123](https://github.com/Ameyabarve123)
- [@cindy-muniz](https://github.com/cindy-muniz)

Licensed under the terms in [`LICENSE`](LICENSE).
