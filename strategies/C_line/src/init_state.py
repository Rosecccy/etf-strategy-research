from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent


def build_state(line: str, source: Path, target: Path, asof: pd.Timestamp) -> dict:
    trades = pd.read_csv(source, dtype={"symbol": str}, encoding="utf-8-sig")
    entry_col = "new_entry_date" if "new_entry_date" in trades.columns else "entry_date"
    exit_col = "new_exit_date" if "new_exit_date" in trades.columns else "exit_date"
    trades[entry_col] = pd.to_datetime(trades[entry_col], errors="coerce")
    trades[exit_col] = pd.to_datetime(trades[exit_col], errors="coerce")
    started = trades[trades[entry_col] <= asof].sort_values(entry_col)

    position = None
    last_exit = None
    last_activity = None
    if not started.empty:
        latest = started.iloc[-1]
        exit_date = latest[exit_col]
        if pd.isna(exit_date) or exit_date > asof:
            position = {
                "symbol": str(latest.get("symbol", "")).zfill(6),
                "display_name": str(latest.get("display_name", "")),
                "entry_date": latest[entry_col].strftime("%Y-%m-%d"),
                "planned_exit_date": exit_date.strftime("%Y-%m-%d") if pd.notna(exit_date) else None,
            }
        completed = started[started[exit_col].notna() & (started[exit_col] <= asof)]
        last_exit = completed[exit_col].max() if not completed.empty else None
        dates = [started[entry_col].max()]
        if pd.notna(last_exit):
            dates.append(last_exit)
        last_activity = max(dates)

    state = {
        "line": line,
        "mode": "paper_model",
        "initialized": True,
        "actual_account_confirmed": False,
        "initialized_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "asof_date": asof.strftime("%Y-%m-%d"),
        "source_file": str(source.relative_to(PROJECT)),
        "last_activity_date": last_activity.strftime("%Y-%m-%d") if pd.notna(last_activity) else None,
        "last_exit_date": last_exit.strftime("%Y-%m-%d") if pd.notna(last_exit) else None,
        "position_status": "holding" if position else "cash",
        "position": position,
        "pending_order": None,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize C and S paper-model state.")
    parser.add_argument("--asof", default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    asof = pd.Timestamp(args.asof)
    jobs = [
        ("C2.4", ROOT / "fit" / "fallback" / "final_trades.csv", ROOT / "account" / "model_state.json"),
        ("S1", PROJECT / "S" / "fit" / "selector" / "final_trades.csv", PROJECT / "S" / "live" / "model_state.json"),
    ]
    result = {}
    for line, source, target in jobs:
        if target.exists() and not args.force:
            result[line] = json.loads(target.read_text(encoding="utf-8"))
        else:
            result[line] = build_state(line, source, target, asof)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
