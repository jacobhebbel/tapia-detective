"""Generate content/rooms.json from House_Generator.py with a fixed seed.

This script is a one-time (well, re-runnable-but-deterministic) content
authoring step. It imports the existing, unmodified `generate_house`
function from the repo-root `House_Generator.py`, calls it with a FIXED
seed so the resulting room graph is reproducible, and writes the result to
`content/rooms.json`.

Seed choice: 1978 — the year "Nautilus" (the tabletop/video game referenced
as a prop/flavor detail associated with the Ammonite fossil in the cast
material) originally shipped. Purely a flavor pick; any fixed integer would
do, since House_Generator.py guarantees a fully connected graph (every room
connects to the "Corridor" hub) for any seed.

Usage:
    python scripts/generate_rooms.py

Run from anywhere; paths are resolved relative to this file, not the CWD.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Repo root is the parent of this script's directory (scripts/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent
CONTENT_DIR = REPO_ROOT / "content"
OUTPUT_PATH = CONTENT_DIR / "rooms.json"

# Fixed seed for reproducibility across every re-run of this script.
FIXED_SEED = 1978

# Make sure `House_Generator.py` (repo root) is importable regardless of CWD.
sys.path.insert(0, str(REPO_ROOT))

from House_Generator import generate_house  # noqa: E402  (import after sys.path tweak)


def main() -> None:
    house = generate_house(seed=FIXED_SEED)

    CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(house, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote {len(house)} rooms (seed={FIXED_SEED}) to {OUTPUT_PATH}")
    for room, neighbors in sorted(house.items()):
        print(f"  {room}: {neighbors}")


if __name__ == "__main__":
    main()
