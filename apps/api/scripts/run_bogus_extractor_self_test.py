from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import _run_bogus_extractor_self_test  # noqa: E402


if __name__ == "__main__":
    _run_bogus_extractor_self_test()
    print("ok")
