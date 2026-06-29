from __future__ import annotations

import runpy
from pathlib import Path


LEGACY_MAIN = Path(__file__).resolve().parents[1] / "Convergence per sweep" / "main.py"


if __name__ == "__main__":
    runpy.run_path(str(LEGACY_MAIN), run_name="__main__")
