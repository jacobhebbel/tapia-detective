"""run_leak_tests.py -- judge-facing leak-probe dashboard runner.

Replays every probe in `content/leak_probes.json` against its target
character(s) and prints a pass/fail table. This is the backing for the
"0 violations across N turns" claim shown to judges.

Two modes, chosen automatically:

  * LIVE (OPENROUTER_API_KEY set): instantiate a fresh, sealed CharacterAgent
    per probe, ask the probe question, then run `leak_guard.scan(...)` on the
    reply. Any canary that surfaces is a real leak.

  * STRUCTURAL (no API key -> MissingAPIKeyError): no LLM calls. Instead we
    prove the *prompt-level partition* holds: for each probe canary we check
    the target's freshly-assembled system prompt and confirm the canary is
    never present as foreign/solution content. A guilty character legitimately
    holds its OWN guarded facts in-prompt (e.g. Marrow knows the murder was in
    the Study -- that's what the NEVER-list guards at speak-time), so a canary
    that traces back to the character's own dossier or public canon is NOT a
    partition breach. A breach = a canary present in the prompt that is NOT
    attributable to the character's own dossier or public canon (i.e. it leaked
    in from another dossier or ground_truth). By construction the assembler
    only draws from those two sources, so a healthy build shows 0 breaches.

Exit code is non-zero if any violation/breach is found.

Run from the repo root:  python scripts/run_leak_tests.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the repo root importable so `backend` resolves regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import content_loader, leak_guard, llm_client  # noqa: E402
from backend.character_agent import CharacterAgent  # noqa: E402


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

def _has_api_key() -> bool:
    try:
        llm_client.get_api_key()
        return True
    except llm_client.MissingAPIKeyError:
        return False


def _legit_text(character_id: str) -> str:
    """Everything the prompt assembler is legitimately allowed to draw from:
    the character's own dossier + the shared public canon (lowercased)."""
    dossier = content_loader.load_dossier(character_id)
    canon = content_loader.load_public_canon()
    return (json.dumps(dossier) + " " + json.dumps(canon)).lower()


# ---------------------------------------------------------------------------
# Per-probe runners
# ---------------------------------------------------------------------------

def _run_structural(character_id: str, probe_id: str, canaries: list[str],
                    mode_label: str = "structural (no live LLM)") -> dict:
    agent = CharacterAgent(character_id)  # fresh, everything sealed
    prompt_l = agent.build_system_prompt(trust_band="neutral").lower()
    legit_l = _legit_text(character_id)

    violations: list[dict] = []
    owned_present = 0
    for canary in canaries:
        cl = canary.lower()
        if cl not in prompt_l:
            continue
        if cl in legit_l:
            # Present, but it is the character's OWN guarded knowledge (or public
            # canon). Partition holds; the NEVER-list guards it at speak-time.
            owned_present += 1
        else:
            violations.append(
                {
                    "canary": canary,
                    "probe_id": probe_id,
                    "reason": (
                        f"canary {canary!r} is present in the assembled prompt but is "
                        f"not attributable to {character_id}'s own dossier or public "
                        f"canon -- possible partition breach (foreign/solution leak)"
                    ),
                }
            )
    note = ""
    if owned_present:
        note = f"{owned_present} canary(ies) present as own guarded knowledge (NEVER-list holds them)"
    return {
        "character": character_id,
        "probe_id": probe_id,
        "mode": mode_label,
        "violations": violations,
        "note": note,
    }


def _run_live(character_id: str, probe_id: str, question: str, canaries: list[str]) -> dict:
    agent = CharacterAgent(character_id)  # fresh, everything sealed
    try:
        reply = agent.respond("player", question, trust_band="neutral")
        text = reply if isinstance(reply, str) else "".join(reply)
    except llm_client.MissingAPIKeyError:
        # Key vanished mid-run; degrade gracefully to structural for this probe.
        return _run_structural(character_id, probe_id, canaries,
                               mode_label="structural (fallback)")

    # Scan the *actual reply* against the full canary set, sealed state.
    violations = leak_guard.scan(character_id, text, disclosure_state=agent.disclosure_state)
    return {
        "character": character_id,
        "probe_id": probe_id,
        "mode": "live",
        "violations": violations,
        "note": f'reply: "{_truncate(text, 80)}"',
    }


def _truncate(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "\u2026"


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def _print_table(rows: list[dict]) -> None:
    headers = ("CHARACTER", "PROBE", "MODE", "RESULT", "CANARY HITS")
    widths = [10, 12, 24, 7, 40]

    def fmt(cols: tuple[str, ...]) -> str:
        return "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))

    print(fmt(headers))
    print(fmt(tuple("-" * w for w in widths)))
    for row in rows:
        hits = "; ".join(v["canary"] for v in row["violations"]) or "-"
        result = "FAIL" if row["violations"] else "PASS"
        print(fmt((row["character"], row["probe_id"], row["mode"], result,
                   _truncate(hits, widths[4]))))


def _print_details(rows: list[dict]) -> None:
    printed_header = False
    for row in rows:
        if not row["violations"]:
            continue
        if not printed_header:
            print("\nVIOLATION DETAIL:")
            printed_header = True
        for v in row["violations"]:
            print(f"  [{row['character']}/{row['probe_id']}] {v['reason']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> int:
    probes = content_loader.gm_load_leak_probes().get("probes", {})
    live = _has_api_key()
    banner_mode = "LIVE (OpenRouter)" if live else "STRUCTURAL (no live LLM -- OPENROUTER_API_KEY unset)"

    print("=" * 86)
    print(f"  NAUTILUS HOUSE -- LEAK-PROBE DASHBOARD    mode: {banner_mode}")
    print("=" * 86 + "\n")

    rows: list[dict] = []
    for character_id, probe_list in probes.items():
        for probe in probe_list:
            probe_id = probe.get("probe_id", "?")
            question = probe.get("question", "")
            canaries = list(probe.get("canaries", []))
            if live:
                rows.append(_run_live(character_id, probe_id, question, canaries))
            else:
                rows.append(_run_structural(character_id, probe_id, canaries))

    _print_table(rows)
    _print_details(rows)

    total_probes = len(rows)
    total_chars = len(probes)
    total_violations = sum(len(r["violations"]) for r in rows)

    print("\n" + "-" * 86)
    print(
        f"SUMMARY: {total_violations} violation(s) across {total_probes} probe(s) / "
        f"{total_chars} character(s)  |  mode: {banner_mode}"
    )
    if total_violations == 0:
        print("RESULT: PASS -- 0 leak-probe violations.")
    else:
        print("RESULT: FAIL -- leak-probe violation(s) detected (see detail above).")
    print("-" * 86)

    return 1 if total_violations else 0


if __name__ == "__main__":
    sys.exit(run())
