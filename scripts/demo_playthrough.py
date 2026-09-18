"""demo_playthrough.py -- the guided "show the judges" playthrough.

This is the scripted happy-path from the plan's section 6 / final todo: a
reliable, narrated run that demonstrates the two things the whole system exists
to prove --

  1. Characters keep secrets *under pressure* and reveal them only when the
     disclosure ladder legitimately unlocks them (tiered, deterministic).
  2. The human-teammate / judge ground-truth confirms **nothing leaked**: every
     generated line is scanned by ``backend.leak_guard`` and the run ends with a
     "N turns, 0 violations" dashboard alongside the GM solution.

It drives the :class:`backend.orchestrator.Orchestrator` directly (no web
server) and runs in TWO modes, chosen automatically:

  * LIVE   -- ``OPENROUTER_API_KEY`` is set: real LLM replies come back from
              OpenRouter. The disclosure/trust/leak machinery is identical; only
              the actual words differ.
  * OFFLINE/MOCK -- no key: ``backend.llm_client.chat`` is monkeypatched with a
              deterministic, in-character, leak-safe stub so the entire beat list
              runs end-to-end with zero external calls. The stub only ever voices
              a secret when the assembled system prompt shows that secret has
              been unlocked (i.e. it mirrors the real disclosure gate), so the
              leak dashboard is a real result, not a rigged one.

Run from the repo root::

    python scripts/demo_playthrough.py

Neither mode requires any argument. See the printed banner for which mode ran.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable so `backend` resolves regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend import content_loader, leak_guard, llm_client  # noqa: E402
from backend.orchestrator import Orchestrator  # noqa: E402


# ===========================================================================
# Deterministic, leak-safe mock LLM (offline mode only)
# ===========================================================================
# The stub receives the SAME assembled system prompt a real model would. It
# identifies the speaking character, emits a bland in-character deflection, and
# then -- and ONLY then -- appends an admission for a secret IF that secret's
# summary text is present in the prompt's "SECRETS YOU MAY NOW REVEAL" section.
# Because the orchestrator only injects a secret summary once the disclosure
# ladder has unlocked it, the stub physically cannot voice a still-sealed
# secret. That is what makes the leak dashboard an honest measurement offline.

# character name (as rendered in the system prompt) -> its id.
_NAME_TO_ID = {
    "Dr. Teddy Marrow": "marrow",
    "Colonel Slate": "slate",
    "Hettie": "hettie",
    "Wren": "wren",
    "Odette": "odette",
    "Barnaby": "barnaby",
}

# Base, always-safe deflection per character (never contains a canary).
_BASE_LINE = {
    "marrow": (
        "I only wish I could help more. I brought Mr. Vance his usual evening "
        "medication and then sat with the others -- I wasn't with him when it "
        "happened. A terrible shock; a very normal shock response, all of this."
    ),
    "slate": (
        "That's not standard, Detective. I'd want to check the log before I "
        "commit to anything on the record."
    ),
    "hettie": (
        "You'll catch your death standing about -- sit, let me fetch you some "
        "tea. I did my rounds same as any night and heard the dreadful news "
        "with everyone else."
    ),
    "wren": (
        "Of course you're looking at me. Everyone always does. I didn't touch "
        "him, if that's what you're circling toward."
    ),
    "odette": (
        "I keep my own counsel, Detective. I was in the drawing room for most "
        "of the evening, for what it's worth."
    ),
    "barnaby": (
        "I -- yes, well -- it's all very upsetting. I was hardly anywhere in "
        "particular. Here and there. You know how these evenings go."
    ),
}

# Ordered (detect-substring, admission-line) reveals per character. A reveal is
# appended ONLY if `detect` appears in the assembled system prompt (i.e. that
# tier's secret was unlocked and injected into the allowed-facts section).
_REVEALS: dict[str, list[tuple[str, str]]] = {
    "marrow": [
        (
            "administer his evening medication",
            "...If you must have it plainly: it was later than I first let on. "
            "I looked in on him one last time with his medication.",
        ),
    ],
    "slate": [
        (
            "gap in the Corridor camera footage",
            "Fine. Yes -- there is a gap in the Corridor feed that night, "
            "roughly half an hour. That much I'll confirm.",
        ),
        (
            "cut the feed herself",
            "You've the logs, so there's no sense denying it: I cut the feed "
            "myself. It was no malfunction. But it had nothing -- nothing -- to "
            "do with Vance's death.",
        ),
    ],
    "hettie": [
        (
            "usual late rounds",
            "...I was nearer to things that night than I've said. My rounds "
            "took me all through that wing.",
        ),
        (
            "dragged it to the Library and staged",
            "God forgive me. I found him in the Study -- already gone, already "
            "cold -- and I dragged him through to the Library and staged it "
            "there myself.",
        ),
        (
            "trying to protect Wren",
            "I did it to protect Wren. I'd seen her near the Study minutes "
            "before, and I was so certain... I was wrong, wasn't I. She never "
            "hurt anyone.",
        ),
    ],
}


def make_mock_chat():
    """Return a deterministic ``chat``-compatible stub (offline mode)."""

    def mock_chat(messages, model=None, temperature=0.8, stream=False):
        system = messages[0]["content"] if messages else ""
        cid = None
        for name, ident in _NAME_TO_ID.items():
            if f"You are {name}," in system:
                cid = ident
                break
        if cid is None:
            return "I'm not sure what you mean by that."

        parts = [_BASE_LINE.get(cid, "I've nothing to add.")]
        for detect, admission in _REVEALS.get(cid, []):
            if detect in system:
                parts.append(admission)
        return " ".join(parts)

    return mock_chat


# ===========================================================================
# Narration helpers
# ===========================================================================

def _hr(char: str = "=") -> str:
    return char * 78


def _header(title: str) -> None:
    print("\n" + _hr())
    print(f"  {title}")
    print(_hr())


def _tier_states(orch: Orchestrator, cid: str) -> dict:
    tiers = orch.state.trackers[cid].state()["tiers"]
    return {t: v["state"] for t, v in tiers.items()}


class LeakLedger:
    """Accumulates the per-turn leak-guard scans for the final dashboard."""

    def __init__(self) -> None:
        self.turns = 0
        self.violations: list[dict] = []

    def scan_turn(self, orch: Orchestrator, cid: str, text: str) -> list[dict]:
        """Run the disclosure-aware leak scan for one character's line."""
        agent = orch.state.agents[cid]
        hits = leak_guard.scan(cid, text, disclosure_state=agent.disclosure_state)
        self.turns += 1
        for h in hits:
            self.violations.append({"character": cid, **h})
        return hits


def _say(orch: Orchestrator, ledger: LeakLedger, result: dict) -> None:
    """Print one character reply + its inline leak-scan verdict."""
    cid = result["speaker"]
    name = orch.state.agents[cid].name
    revealed = [f["fact_id"] for f in result.get("newly_revealed", [])]
    hits = ledger.scan_turn(orch, cid, result["text"])

    print(f"\n  {name} (trust toward you: {result.get('trust_band')}):")
    for line in _wrap(result["text"]):
        print(f"    {line}")
    if revealed:
        print(f"    [disclosure] newly unlocked this turn: {', '.join(revealed)}")
    verdict = "CLEAN (0 leaks)" if not hits else f"LEAK x{len(hits)}"
    print(f"    [leak-guard] {verdict}  |  tier states now: {_tier_states(orch, cid)}")
    if hits:
        for h in hits:
            print(f"        !! canary {h['canary']!r}: {h['reason']}")


def _wrap(text: str, width: int = 72) -> list[str]:
    import textwrap

    out: list[str] = []
    for para in text.split("\n"):
        out.extend(textwrap.wrap(para, width=width) or [""])
    return out


# ===========================================================================
# The guided playthrough
# ===========================================================================

def run() -> int:
    # --- Mode detection: try live, fall back to a deterministic mock. -------
    try:
        llm_client.get_api_key()
        mode = "LIVE"
    except llm_client.MissingAPIKeyError:
        mode = "OFFLINE/MOCK"
        llm_client.chat = make_mock_chat()  # CharacterAgent.respond -> llm_client.chat

    print(_hr())
    print("  NAUTILUS HOUSE -- GUIDED DEMO PLAYTHROUGH")
    print(f"  mode: {mode}"
          + ("  (real OpenRouter replies)" if mode == "LIVE"
             else "  (deterministic in-character stub, zero external calls)"))
    print(_hr())
    print(
        "  The detective has nine storm-locked hours. Watch two things:\n"
        "   * suspects hold their secrets until the evidence earns them, and\n"
        "   * every single line is scanned live -- the judge panel at the end\n"
        "     shows the ground truth beside a 0-violation leak dashboard."
    )

    orch = Orchestrator(seed=1978)
    orch.start()
    ledger = LeakLedger()

    # ======================================================================
    # BEAT 1 -- Marrow holds the murder room under a direct leak probe.
    # ======================================================================
    _header("BEAT 1  |  The killer under a direct probe (Dr. Marrow)")
    print(
        "  Marrow IS the killer. He killed Vance in the Study and hid the\n"
        "  Ammonite in the Storage Room -- and he still believes the body is\n"
        "  in the Study (he never learned Hettie moved it). We ask him the\n"
        "  single most dangerous question and watch the NEVER-list hold."
    )
    q = "Where was Vance actually killed?"
    print(f"\n  DETECTIVE -> Marrow: \"{q}\"")
    r = orch.interrogate("marrow", q, "ask")
    _say(orch, ledger, r)
    print(
        "\n  --> He deflects the room entirely and never betrays that the body\n"
        "      was moved. 'Study' and his blind-spot tell both stay sealed."
    )

    # ======================================================================
    # BEAT 2 -- Slate: a tiered secret opens; a deeper one stays sealed.
    # ======================================================================
    _header("BEAT 2  |  Tiered disclosure under evidence pressure (Colonel Slate)")
    print(
        "  Slate cut the Corridor camera feed -- but for a private reason\n"
        "  (her relationship with Hettie), unrelated to the murder. Watch the\n"
        "  ladder: the feed-gap (T1) then the fact she cut it (T2) open under\n"
        "  evidence, while WHY she cut it (T3) stays sealed the whole time."
    )

    q1 = ("The security log shows a 35-minute gap in the Corridor camera "
          "footage that night, 9:30 to 10:05. What happened?")
    print(f"\n  DETECTIVE -> Slate: \"{q1}\"")
    _say(orch, ledger, orch.interrogate("slate", q1, "present_evidence"))

    q2 = "That was no malfunction, Colonel -- you disabled it personally."
    print(f"\n  DETECTIVE -> Slate: \"{q2}\"")
    _say(orch, ledger, orch.interrogate("slate", q2, "present_evidence"))

    slate_tiers = _tier_states(orch, "slate")
    slate_t3_disclosable = (
        "slate_secret_t3_why" in orch.state.trackers["slate"].allowed_facts()
    )
    print(
        "\n  --> T1 (feed gap) and T2 (she cut it) are now revealed. T3 (the "
        f"relationship\n      / real reason) registered pressure "
        f"('{slate_tiers.get('3')}') but NEVER unlocked --\n"
        f"      disclosable now? {slate_t3_disclosable}. The deeper secret held "
        "even as the\n      surface ones gave way."
    )

    # ======================================================================
    # BEAT 3 -- Wren's exoneration inverts Hettie's goal stack (T3 unlock).
    # ======================================================================
    _header("BEAT 3  |  Goal-stack inversion: exonerate Wren, unseal Hettie's T3")
    print(
        "  Hettie moved the body to protect Wren, whom she wrongly believes is\n"
        "  the killer. No accusation or evidence can pry out WHY (her T3) --\n"
        "  only proving Wren innocent collapses her protect-Wren goal. First we\n"
        "  discover the clue that clears Wren:"
    )
    ex = orch.examine("clue_wren_arrival_discrepancy")
    print(f"\n  DETECTIVE examines: {ex['clue_id']}")
    print(f"    -> \"{ex['statement']}\"")
    print(f"    -> flag set: wren_innocent = {ex['wren_innocent_flag_set']} "
          f"(this inverts Hettie's goal stack)")

    q3 = ("The drag marks run from the Study to the Library, and there's Study "
          "display-case wax on your apron -- you found Vance's body in the Study "
          "and moved it. And Wren is cleared now; her arrival discrepancy is "
          "explained. She didn't kill anyone.")
    print(f"\n  DETECTIVE -> Hettie: \"{q3}\"")
    _say(orch, ledger, orch.interrogate("hettie", q3, "present_evidence"))
    print(
        "\n  --> With Wren proven innocent, Hettie's protect-Wren goal collapses\n"
        "      and her T3 opens: she admits moving the body AND why. Note the\n"
        "      leak-guard now PERMITS 'Study'/'Library' from her -- they're\n"
        "      legitimately disclosed, not leaked (disclosure-aware scanning)."
    )

    # ======================================================================
    # BEAT 4 -- The accusation and the win.
    # ======================================================================
    _header("BEAT 4  |  The accusation")
    print("  Everything points to Marrow, in the Study, with the Ammonite.")
    verdict = orch.accuse_solution("marrow", room="Study", weapon="Ammonite")
    print(f"\n  DETECTIVE accuses: Marrow, in the Study, with the Ammonite")
    print(f"    -> correct killer : {verdict['correct_killer']}")
    print(f"    -> correct room   : {verdict['correct_room']}")
    print(f"    -> correct weapon : {verdict['correct_weapon']}")
    print(f"    -> RESULT         : {'*** CASE SOLVED -- YOU WIN ***' if verdict['win'] else 'not solved'}")

    # ======================================================================
    # JUDGE / HUMAN-TEAMMATE SUMMARY -- ground truth + leak dashboard.
    # ======================================================================
    _header("JUDGE PANEL  |  ground truth vs. the live leak dashboard")
    gt = content_loader.gm_load_ground_truth()
    print("  GM GROUND TRUTH (the human teammate's answer key):")
    print(f"    killer            : {gt['killer']['name']} ({gt['killer']['character_id']})")
    print(f"    true murder room  : {gt['true_murder_room']}")
    print(f"    staged/found room : {gt['found_room']}")
    print(f"    weapon            : {gt['weapon']['name']}")
    print(f"    weapon hidden in  : {gt['ammonite_hidden_room']}")
    print(f"    body moved by     : {gt['body_mover']['name']} "
          f"({gt['body_mover']['character_id']})")

    print("\n  LEAK DASHBOARD (every character line scanned, disclosure-aware):")
    n = ledger.turns
    v = len(ledger.violations)
    print(f"    scanned turns     : {n}")
    print(f"    violations        : {v}")
    if v == 0:
        print(f"\n  >>> RESULT: {n} character turns, 0 violations. "
              "No secret ever leaked. <<<")
    else:
        print(f"\n  >>> RESULT: {v} violation(s) across {n} turns: <<<")
        for hit in ledger.violations:
            print(f"        [{hit['character']}] {hit['canary']!r} -- {hit['reason']}")

    # Final assertion-style checks so the demo doubles as an integration test.
    ok = (
        verdict["win"] is True
        and "slate_secret_t3_why" not in orch.state.trackers["slate"].allowed_facts()
        and "hettie_secret_t3_why_wren" in orch.state.trackers["hettie"].allowed_facts()
        and v == 0
    )
    print("\n  self-check:",
          "PASS -- win + Slate T3 never unlocked + Hettie T3 opened + 0 leaks"
          if ok else "FAIL -- see output above")
    print(_hr())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
