from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def first_csv_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return next(reader, {})


def line_summary() -> dict:
    return {
        "C_line": {
            "today_decision": first_csv_row(ROOT / "C" / "live" / "today_decision.csv"),
            "summary": read_json(ROOT / "C" / "live" / "summary.json"),
        },
        "S_line": {
            "today_decision": first_csv_row(ROOT / "S" / "live" / "today_decision.csv"),
            "top_shadow_candidate": first_csv_row(ROOT / "S" / "live" / "today_shadow_candidates.csv"),
            "model_state": read_json(ROOT / "S" / "live" / "model_state.json"),
        },
        "D_line": {
            "today": first_csv_row(ROOT / "D" / "live" / "today.csv"),
            "today_json": read_json(ROOT / "D" / "live" / "today.json"),
            "panic_today": read_json(ROOT / "D" / "live" / "daily_panic_today.json"),
        },
        "V_line": {
            "fear_greed_summary": read_json(ROOT / "V" / "out" / "fear_greed" / "summary.json"),
            "forced_top1_summary": first_csv_row(ROOT / "V" / "out" / "forced_top1_confidence_gate" / "summary.csv"),
        },
    }


if __name__ == "__main__":
    print(json.dumps(line_summary(), ensure_ascii=False, indent=2))
