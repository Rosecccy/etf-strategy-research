from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REQUIRED = [
    ROOT / "C" / "raw" / "etf",
    ROOT / "C" / "live" / "today_decision.csv",
    ROOT / "S" / "raw" / "etf",
    ROOT / "S" / "live" / "today_decision.csv",
    ROOT / "D" / "raw" / "etf",
    ROOT / "D" / "live" / "today.csv",
    ROOT / "V" / "out" / "fear_greed",
    ROOT / "run_all_strategies.py",
]


def main() -> int:
    bootstrap = ROOT / "scripts" / "make_repo_runnable.py"
    if bootstrap.exists():
        subprocess.run([sys.executable, str(bootstrap)], cwd=ROOT, check=True)

    missing = [str(p.relative_to(ROOT)) for p in REQUIRED if not p.exists()]
    if missing:
        print("PACKAGE NOT RUNNABLE")
        print(json.dumps({"missing": missing}, ensure_ascii=False, indent=2))
        return 1

    result = subprocess.run(
        [sys.executable, str(ROOT / "run_all_strategies.py")],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )
    if result.returncode != 0:
        print("RUNNER FAILED")
        print(result.stdout)
        print(result.stderr)
        return result.returncode

    print("PACKAGE RUNNABLE")
    print(result.stdout)

    deep_checks = [
        ["C", [sys.executable, str(ROOT / "C" / "src" / "run_daily.py"), "--skip-update"]],
        ["S", [sys.executable, str(ROOT / "S" / "src" / "run_s1_live.py")]],
        ["D", [sys.executable, str(ROOT / "D" / "src" / "daily_panic_live.py")]],
    ]
    for name, cmd in deep_checks:
        check = subprocess.run(cmd, cwd=ROOT, text=True, encoding="utf-8", capture_output=True)
        if check.returncode != 0:
            print(f"{name}_LINE_ENTRY_FAILED")
            print(check.stdout)
            print(check.stderr)
            return check.returncode
        print(f"{name}_LINE_ENTRY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
