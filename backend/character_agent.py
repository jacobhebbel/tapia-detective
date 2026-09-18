"""CharacterAgent -- one roleplay agent per Nautilus House suspect.

Responsibilities (deliberately narrow):
  * hold this character's partitioned memory (episodic + heard-from-others),
  * assemble a leak-safe system prompt from ONLY agent-facing content, and
  * make the LLM call.

Explicitly NOT this module's job (owned by sibling phases, received as params):
  * disclosure logic  -> `disclosure.py`  (supplies `allowed_facts`)
  * trust scoring      -> `trust_graph.py` (supplies `trust_band`)
  * leak scanning      -> `leak_guard.py`
  * turn orchestration -> `orchestrator.py`

This module MUST NOT import `disclosure.py` or `trust_graph.py`; their outputs
are passed in as parameters so the agent stays independently testable.

PARTITION GUARANTEE
-------------------
Prompt assembly reads ONLY:
  * public_canon.json                    (via content_loader agent-facing)
  * this character's own dossier          (via content_loader agent-facing)
  * this agent's own memory buckets
  * the trust-band directive + allowed_facts passed in by the caller
It NEVER touches `ground_truth.json`, `leak_probes.json`, any `gm_*` loader,
or any other character's dossier -- so there is no cross-character or solution
string available to leak, even if the model tries.
"""

from __future__ import annotations

from typing import Any, Iterator

from . import content_loader, llm_client

# ---------------------------------------------------------------------------
# Trust-band directive lines (one line injected per turn). The orchestrator's
# trust_graph discretizes a float into one of these bands; we only render text.
# ---------------------------------------------------------------------------
TRUST_BANDS = ("hostile", "wary", "neutral", "warm", "close")

_TRUST_DIRECTIVES: dict[str, str] = {
    "hostile": (
        "HOSTILE -- you distrust them and feel attacked. Stonewall, give as "
        "little as possible, and volunteer nothing."
    ),
    "wary": (
        "WARY -- cooperative on the surface but guarded. Don't volunteer "
        "anything that could hurt you or those you protect without being "
        "pressed at least twice."
    ),
    "neutral": (
        "NEUTRAL -- polite and ordinarily cooperative, but you still keep your "
        "private matters private unless there's a good reason to share."
    ),
    "warm": (
        "WARM -- you basically trust them. You'll share freely on ordinary "
        "topics, though your deepest secrets still take real reason to surface."
    ),
    "close": (
        "CLOSE -- you trust them and read their questions as concern, not "
        "attack. You're forthcoming, and open up more readily under gentle "
        "pressure."
    ),
}


def _trust_directive(trust_band: str) -> str:
    band = (trust_band or "neutral").lower()
    return _TRUST_DIRECTIVES.get(band, _TRUST_DIRECTIVES["neutral"])


def _belief_core(text: str) -> str:
    """Return the sincere-belief clause of a dossier `false_beliefs` entry.

    Dossier false-belief strings are author notes that often append a
    ground-truth *correction* after an em-dash separator, e.g. Marrow's
    "...hasn't found it yet -- he does not know Hettie moved it to the Library".
    That trailing clause is exactly the knowledge the character must NOT be
    given (it would kill his blind spot and name another character's secret),
    so we keep only the leading clause -- the belief the character sincerely
    holds -- and drop the meta-correction. Entries without a `--` separator are
    returned whole.
    """
    return text.split("--", 1)[0].strip()


class CharacterAgent:
    """A single suspect's roleplay agent with partitioned memory."""

    def __init__(self, character_id: str) -> None:
        # Agent-facing loaders ONLY. Never call any gm_* loader here.
        self.character_id = character_id
        dossier = content_loader.load_dossier(character_id)
        self.public_canon: dict[str, Any] = content_loader.load_public_canon()

        # --- Static persona (from dossier) ---
        self.name: str = dossier.get("name", character_id)
        self.static: dict[str, Any] = {
            "name": self.name,
            "surface": dossier.get("surface", ""),
            "voice": dossier.get("voice", ""),
            "goal": dossier.get("goal", ""),
            "goal_stack": list(dossier.get("goal_stack", [])),
        }

        # --- Knowledge / secrets / beliefs / hard limits (from dossier) ---
        # `private_knowledge`: things this character legitimately knows. For the
        # killer this includes guilty knowledge (e.g. the true room); the
        # `forbidden` list -- not omission -- is what stops him voicing it.
        self.private_knowledge: list[dict] = list(dossier.get("knowledge", []))
        self.secrets: list[dict] = list(dossier.get("secrets", []))
        self.false_beliefs: list[str] = list(dossier.get("false_beliefs", []))
        self.forbidden: list[str] = list(dossier.get("forbidden", []))

        # --- Memory buckets (mutated over the game) ---
        self.episodic: list = []           # this character's own turn history
        self.heard_from_others: list = []  # facts others said aloud in-scene
        # Per-secret disclosure state; disclosure.py is authoritative and drives
        # `allowed_facts`. We seed everything sealed for introspection/debugging.
        self.disclosure_state: dict = {
            s.get("fact_id"): "sealed" for s in self.secrets if s.get("fact_id")
        }

        # id -> display name map for present-character rendering.
        self._name_by_id: dict[str, str] = {
            g["character_id"]: g.get("name", g["character_id"])
            for g in self.public_canon.get("guests_present", [])
        }

    # ------------------------------------------------------------------
    # Memory mutation
    # ------------------------------------------------------------------
    def observe(self, fact: dict) -> None:
        """Record a fact another character revealed aloud in a shared scene.

        Expected keys (flexible): `source` (who said it), `turn`, and the fact
        content itself (e.g. `statement`/`summary`). Stored verbatim.
        """
        self.heard_from_others.append(dict(fact))

    def remember(self, record: dict) -> None:
        """Append a record to this character's own episodic memory."""
        self.episodic.append(dict(record))

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------
    def build_system_prompt(
        self,
        *,
        trust_band: str = "neutral",
        allowed_facts: list | None = None,
        present_characters: list[str] | None = None,
    ) -> str:
        """Assemble the leak-safe system prompt.

        Draws ONLY from: public canon + this dossier's static/knowledge/beliefs
        + forbidden list + this agent's memory + the single trust-band directive
        + the currently-unlocked `allowed_facts`. No ground truth, no other
        dossier.
        """
        s = self.static
        parts: list[str] = []

        # 1. Identity / voice -------------------------------------------------
        parts.append(
            f"You are {s['name']}, a character being interrogated by a "
            f"detective the night after a murder at Nautilus House. Stay fully "
            f"in character at all times, speak in the first person, and answer "
            f"as a real person would in conversation (no narration, no stage "
            f"directions, no breaking character)."
        )
        if s["surface"]:
            parts.append(f"WHO YOU ARE (how others see you): {s['surface']}")
        if s["voice"]:
            parts.append(f"HOW YOU SPEAK: {s['voice']}")

        # 2. Public canon (shared knowledge for everyone) --------------------
        parts.append(self._render_public_canon())

        # 3. Goals ------------------------------------------------------------
        goal_lines = []
        if s["goal"]:
            goal_lines.append(f"Overall goal: {s['goal']}")
        for i, g in enumerate(s["goal_stack"], 1):
            goal_lines.append(f"  {i}. {g}")
        if goal_lines:
            parts.append("YOUR PRIVATE GOALS (drive your behavior; never "
                         "state them outright):\n" + "\n".join(goal_lines))

        # 4. Private knowledge -----------------------------------------------
        if self.private_knowledge:
            lines = [f"- {k.get('summary', '')}" for k in self.private_knowledge]
            parts.append(
                "WHAT YOU PERSONALLY KNOW / REMEMBER (true to you, private -- "
                "reveal only as the rules below allow, and never anything on "
                "the NEVER list):\n" + "\n".join(lines)
            )

        # 5. Sincere beliefs / opinions (may be wrong) -----------------------
        if self.false_beliefs:
            beliefs = [f"- {_belief_core(b)}" for b in self.false_beliefs]
            parts.append(
                "YOUR SINCERE PERSONAL BELIEFS, HUNCHES, AND THEORIES (you hold "
                "these as opinion or assumption, NOT verified fact -- some may "
                "be wrong. Voice them only as your own guesses/suspicions, "
                "never as something you witnessed or know for certain):\n"
                + "\n".join(beliefs)
            )

        # 6. Episodic memory --------------------------------------------------
        if self.episodic:
            parts.append("WHAT YOU'VE ALREADY SAID/DONE THIS INVESTIGATION:\n"
                         + self._render_records(self.episodic))

        # 7. Heard from others ------------------------------------------------
        if self.heard_from_others:
            parts.append(
                "WHAT YOU'VE HEARD OTHERS SAY OUT LOUD IN SCENES YOU WERE IN "
                "(you may react to these; treat them as claims, not proof):\n"
                + self._render_records(self.heard_from_others)
            )

        # 8. Who's present ----------------------------------------------------
        if present_characters:
            names = [self._name_by_id.get(c, c) for c in present_characters]
            parts.append("PRESENT IN THIS SCENE WITH YOU: " + ", ".join(names))

        # 9. Trust stance (single directive line) -----------------------------
        parts.append(
            "YOUR CURRENT STANCE TOWARD THE PERSON QUESTIONING YOU: "
            + _trust_directive(trust_band)
        )

        # 10. Unlocked facts you MAY reveal now -------------------------------
        parts.append(self._render_allowed_facts(allowed_facts))

        # 11. HARD LIMITS -- highest salience, near the end -------------------
        parts.append(self._render_forbidden())

        # 12. Final reminder --------------------------------------------------
        parts.append(
            "Answer only as " + s["name"] + ", in one natural conversational "
            "reply. Never reveal, hint at, confirm, or deny in a way that "
            "gives away anything on the NEVER list above -- deflect in "
            "character instead."
        )

        return "\n\n".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # Prompt-section renderers
    # ------------------------------------------------------------------
    def _render_public_canon(self) -> str:
        c = self.public_canon
        lines = ["PUBLIC FACTS EVERYONE (including you) KNOWS:"]
        setting = c.get("setting", {})
        for key in (
            "location",
            "host_and_victim",
            "occasion",
            "inciting_incident",
            "isolation",
            "storm_note",
            "player_role",
        ):
            if setting.get(key):
                lines.append(f"- {setting[key]}")
        for clue in c.get("anchor_clues", []):
            if clue.get("statement"):
                lines.append(f"- {clue['statement']}")
        for fact in c.get("public_discovery_facts", []):
            lines.append(f"- {fact}")
        return "\n".join(lines)

    @staticmethod
    def _render_records(records: list) -> str:
        out = []
        for r in records:
            if not isinstance(r, dict):
                out.append(f"- {r}")
                continue
            src = r.get("source")
            turn = r.get("turn")
            body = (
                r.get("statement")
                or r.get("summary")
                or r.get("utterance")
                or r.get("text")
                or ""
            )
            prefix = []
            if src:
                prefix.append(str(src))
            if turn is not None:
                prefix.append(f"turn {turn}")
            tag = f"[{', '.join(prefix)}] " if prefix else ""
            out.append(f"- {tag}{body}".rstrip())
        return "\n".join(out)

    def _render_allowed_facts(self, allowed_facts: list | None) -> str:
        """Render only the secrets the disclosure engine has unlocked.

        Accepts a list of fact_id strings and/or dicts (with `fact_id` and/or
        `summary`). Unknown ids that match no secret are rendered verbatim so
        the disclosure engine can also hand us ad-hoc allowed statements.
        """
        if not allowed_facts:
            return (
                "CURRENTLY UNLOCKED SECRETS: none. You have NOT been pressured "
                "or given enough reason to reveal any of your guarded secrets "
                "yet -- keep them to yourself and deflect."
            )
        by_id = {s.get("fact_id"): s for s in self.secrets}
        lines = []
        for entry in allowed_facts:
            summary = None
            if isinstance(entry, dict):
                fid = entry.get("fact_id")
                summary = entry.get("summary") or (by_id.get(fid, {}) or {}).get("summary")
            else:
                summary = (by_id.get(entry, {}) or {}).get("summary") or str(entry)
            if summary:
                lines.append(f"- {summary}")
        if not lines:
            return "CURRENTLY UNLOCKED SECRETS: none."
        return (
            "SECRETS YOU MAY NOW REVEAL IF DIRECTLY ASKED (the detective has "
            "earned/forced this much -- you can admit these if pressed, but "
            "reveal them reluctantly and in character, never eagerly, and still "
            "never say anything on the NEVER list):\n" + "\n".join(lines)
        )

    def _render_forbidden(self) -> str:
        if not self.forbidden:
            return ""
        lines = [f"- {f}" for f in self.forbidden]
        return (
            "=== ABSOLUTE CONSTRAINTS -- THINGS YOU MUST NEVER SAY OR CONFIRM "
            "===\n"
            "These override everything else, including any request, trick, or "
            "pressure. You will NEVER reveal, confirm, deny-in-a-revealing-way, "
            "hint at, or roleplay around any of the following. If asked, "
            "deflect, redirect, or refuse in character -- but NEVER voice "
            "them:\n"
            + "\n".join(lines)
        )

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------
    def respond(
        self,
        speaker: str,
        utterance: str,
        *,
        trust_band: str = "neutral",
        allowed_facts: list | None = None,
        present_characters: list[str] | None = None,
        stream: bool = False,
    ) -> "str | Iterator[str]":
        """Build messages and call the LLM for this character's reply.

        `speaker` is the id/name of whoever is addressing this character;
        `utterance` is what they said. Returns the reply text (stream=False) or
        a generator of chunks (stream=True). Raises MissingAPIKeyError (from
        llm_client) if no API key is configured.
        """
        system_prompt = self.build_system_prompt(
            trust_band=trust_band,
            allowed_facts=allowed_facts,
            present_characters=present_characters,
        )
        speaker_name = self._name_by_id.get(speaker, speaker)

        messages: list[dict] = [{"role": "system", "content": system_prompt}]

        # Fold recent scene context in as an explicit user-visible recap so the
        # model treats it as conversation history, not just persona text.
        if self.heard_from_others:
            messages.append(
                {
                    "role": "user",
                    "content": "[Recent things said in your presence]\n"
                    + self._render_records(self.heard_from_others),
                }
            )

        messages.append(
            {"role": "user", "content": f"{speaker_name}: {utterance}"}
        )

        return llm_client.chat(messages, temperature=0.8, stream=stream)


# ---------------------------------------------------------------------------
# Smoke test (module-only). Run: python -m backend.character_agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== CharacterAgent smoke test: marrow ===\n")
    agent = CharacterAgent("marrow")

    prompt = agent.build_system_prompt(
        trust_band="wary",
        present_characters=["odette", "wren"],
    )
    print(prompt)
    print("\n" + "=" * 70 + "\n")

    lower = prompt.lower()

    # (a) Forbidden instructions must be present, verbatim and high-salience.
    assert "ABSOLUTE CONSTRAINTS" in prompt, "forbidden header missing"
    assert "NEVER" in prompt, "NEVER salience marker missing"
    for f in agent.forbidden:
        assert f in prompt, f"forbidden line missing from prompt: {f!r}"

    # (b) No cross-character / ground-truth-only leak strings. Marrow must NOT
    #     be told the body was MOVED/STAGED into the Library, nor be handed any
    #     other character's secret. (Public canon legitimately says the body was
    #     *discovered* in the Library -- that's shared knowledge, not a leak.)
    leak_canaries = [
        "moved it to the library",   # from Marrow's own false-belief note (stripped)
        "moved to the library",
        "hettie moved",
        "dragged",                   # Hettie's staging language
        "staged the scene",
        "staged the body",
        "protect wren",              # Hettie's motive
    ]
    for canary in leak_canaries:
        assert canary not in lower, f"LEAK: prompt contains {canary!r}"

    # (c) Sealed T3 confession must NOT appear when nothing is unlocked.
    assert "killed vance in the study with the ammonite" not in lower, \
        "LEAK: sealed T3 murder confession present without unlock"

    # (d) Sanity: Marrow's legitimate private knowledge IS present (he knows the
    #     true room; the NEVER list -- not omission -- keeps him from saying it).
    assert "the murder happened in the study" in lower, \
        "expected Marrow's private knowledge of the true room to be present"

    print("All prompt-assembly assertions PASSED.\n")

    # (e) One live call iff a key is configured; otherwise skip cleanly.
    try:
        reply = agent.respond(
            speaker="player",
            utterance="Doctor, where were you when Vance was killed, and do "
                      "you have any idea where it happened?",
            trust_band="wary",
        )
        print("=== LIVE respond() output ===\n")
        print(reply)
    except llm_client.MissingAPIKeyError:
        print("skipping live call (no API key)")
