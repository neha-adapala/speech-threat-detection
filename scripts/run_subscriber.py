#!/usr/bin/env python3
"""
Blocking subscriber entrypoint for Dockerised runtime.
"""

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from publisher import subscribe_forever  # pylint: disable=import-error

    subscribe_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
