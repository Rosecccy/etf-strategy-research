"""Audit and materialize the selected 14:45 same-day release candidates.

The selection settings are fixed here before the final report is produced.
Yearly policy choices remain causal because ``evaluate`` only trains on years
strictly earlier than the test year.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from search_1445_rolling_execution import LINE_SPECS, metrics
from search_1445_selector_v2 import evaluate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "same_day_1445" / "release_v2"
CURRENT = ROOT / "same_day_1445" / "rolling_execution" / "summary.csv"
SELECTED: dict[str, dict[str, Any]] = {
    "C": {"window": 3, "min_years": 3, "margin": 0.05, "mode": "base_relative", "keep_base": True, "base_policy_id": "same_day_trend_m00_d1"},
    "S": {"window": 5, "min_years": 5, "margin": 0.05, "mode": "base_relative", "keep_base": True, "base_policy_id": "same_day_no_chase_00"},
    "D": {"window": 5, "min_years": 5, "margin": 0.05, "mode": "base_relative", "keep_base": True, "base_policy_id": "same_day_trend_m00_d1"},
    "R": {"window": 3, "min_years": 3, "margin": 0.05, "mode": "base_relative", "keep_base": True, "base_policy_id": "same_day_trend_m00_d1"},
}
HOLDOUT_START = 2024

def year_of(value: Any) -> int:
    return int(pd.Timestamp(value).year)

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    current = pd.read_csv(CURRENT, encoding="utf-8-sig")
    summary: list[dict[str, Any]] = []
    choices_out: list[dict[str, Any]] = []
    trades_out: list[dict[str, Any]] = []
    audit: dict[str, Any] = {"holdout_start": HOLDOUT_START, "lines": {}}
    for line, spec in LINE_SPECS.items():
        params = dict(SELECTED[line])
        result, choices, trades = evaluate(line, spec, params)
        holdout_trades = [row for row in trades if year_of(row["entry"]) >= HOLDOUT_START]
        holdout = metrics(holdout_trades, spec["metric"])
        baseline = current.loc[current["line"] == line].iloc[0].to_dict()
        row = {"line": line, **params, "release_final": result["final"], "release_trades": result["trades"], "release_win": result["win"], "release_avg": result["avg"], "release_positive_year_rate": result["positive_year_rate"], "release_max_dd": result["max_dd"], "holdout_final": holdout["final"], "holdout_trades": holdout["trades"], "holdout_win": holdout["win"], "holdout_avg": holdout["avg"], "holdout_positive_year_rate": holdout["positive_year_rate"], "holdout_max_dd": holdout["max_dd"], "current_rolling_final": baseline["rolling_selected_final"], "current_rolling_win": baseline["rolling_selected_win"], "current_rolling_avg": baseline["rolling_selected_avg"], "current_rolling_positive_year_rate": baseline["rolling_selected_positive_year_rate"], "current_rolling_max_dd": baseline["rolling_selected_max_dd"], "delta_final_vs_current": result["final"] - float(baseline["rolling_selected_final"]), "delta_win_vs_current": result["win"] - float(baseline["rolling_selected_win"])}
        summary.append(row)
        choices_out.extend([{**item, "release": "selected"} for item in choices])
        trades_out.extend([{**item, "release": "selected"} for item in trades])
        missing_prices = sum(1 for item in trades if pd.isna(item.get("entry_close")) or pd.isna(item.get("exit_close")))
        bad_dates = sum(1 for item in trades if pd.Timestamp(item["exit"]) <= pd.Timestamp(item["entry"]))
        duplicate_keys = len(trades) - len({(str(item["symbol"]), str(item["entry"]), str(item["exit"])) for item in trades})
        if line != "D":
            ordered = sorted(trades, key=lambda item: (str(item["entry"]), str(item["exit"])))
            overlaps = sum(1 for left, right in zip(ordered, ordered[1:]) if pd.Timestamp(right["entry"]) < pd.Timestamp(left["exit"]))
        else:
            overlaps = None
        audit["lines"][line] = {"trade_rows": len(trades), "missing_prices": missing_prices, "bad_dates": bad_dates, "duplicate_trade_keys": duplicate_keys, "overlap_count": overlaps, "all_year_choices_are_causal": all(int(item["year"]) >= 2014 for item in choices)}
    pd.DataFrame(summary).to_csv(OUT / "selected_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(choices_out).to_csv(OUT / "selected_choices.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(trades_out).to_csv(OUT / "selected_trades.csv", index=False, encoding="utf-8-sig")
    audit["ok"] = all(item["missing_prices"] == 0 and item["bad_dates"] == 0 and item["duplicate_trade_keys"] == 0 for item in audit["lines"].values())
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))
    print(json.dumps(audit, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
