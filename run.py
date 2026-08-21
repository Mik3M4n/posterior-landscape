"""Run posterior-landscape directly from an editable source checkout."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from posterior_landscape.__main__ import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
