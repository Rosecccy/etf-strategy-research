from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..engine.orchestrator import run_optimizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    args = parser.parse_args()
    print(json.dumps(run_optimizer(args.root, args.as_of), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
