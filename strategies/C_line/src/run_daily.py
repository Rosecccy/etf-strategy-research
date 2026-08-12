from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(args: list[str], skip: bool = False) -> dict:
    if skip:
        return {"cmd": " ".join(args), "skipped": True, "returncode": 0}
    proc = subprocess.run(args, cwd=ROOT.parent, capture_output=True, text=True)
    return {
        "cmd": " ".join(args),
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run C2 daily workflow.")
    parser.add_argument("--skip-update", action="store_true", help="Only generate live decision from local data.")
    parser.add_argument("--force-full", action="store_true", help="Force full ETF data refresh.")
    parser.add_argument("--cash", type=float, default=None, help="Optional cash for account reconcile.")
    args = parser.parse_args()

    py = sys.executable
    steps = []
    update_cmd = [py, "C/src/update_data.py"]
    if args.force_full:
        update_cmd.append("--force-full")
    steps.append(run_step(update_cmd, skip=args.skip_update))
    steps.append(run_step([py, "C/src/extrema_live.py"]))
    steps.append(run_step([py, "C/src/live_signal.py"]))
    reconcile_cmd = [py, "C/src/account.py", "reconcile"]
    if args.cash is not None:
        reconcile_cmd += ["--cash", str(args.cash)]
    steps.append(run_step(reconcile_cmd))

    out = {"steps": steps}
    path = ROOT / "live" / "daily_run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
