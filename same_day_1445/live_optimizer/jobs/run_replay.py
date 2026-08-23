from __future__ import annotations

import argparse
from pathlib import Path

from ..engine.orchestrator import run_optimizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    args = parser.parse_args()
    result = run_optimizer(args.root, args.as_of)
    print(result["decision_hash"])


if __name__ == "__main__":
    main()
