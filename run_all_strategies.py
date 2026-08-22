"""Print summaries from the historical C/S/D/V compatibility runtime.

This is not the verifier for the current C/S/D/R V2 baseline. Use
`scripts/verify_current_baseline.py` for the current formal release.
"""

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
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def first_csv_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.DictReader(handle), {})


def line_summary() -> dict:
    return {
        "scope": "legacy_C_S_D_V_compatibility_only",
        "current_baseline_verifier": "scripts/verify_current_baseline.py",
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
