"""Convenience launcher for the Nautilus House FastAPI backend.

Equivalent to::

    uvicorn backend.app:app --reload --port 8000

Run from the repo root::

    python scripts/run_server.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is importable so `backend.app` resolves when this script
# is run directly (python scripts/run_server.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
