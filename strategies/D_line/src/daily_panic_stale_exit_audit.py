from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import daily_panic_rolling_test as base
from control_panic_age_test import portfolio_metrics
from daily_panic_stale_exit_test import RULES, adjust_exits, build_formal


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "out" / "daily_panic_stale_exit_test"
OUT = SOURCE / "audit"


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main() -> None:
    configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    choices = pd.read_csv(SOURCE / "winner_choices.csv", encoding="utf-8-sig")
    expected = pd.read_csv(
        SOURCE / "winner_holdout_trades.csv",
        dtype={"symbol": str},
        encoding="utf-8-sig",
    )
    expected["symbol"] = expected["symbol"].astype(str).str.zfill(6)
    adjusted, raw, pool = base.prepare()
    formal = build_formal(adjusted, raw, pool)
    rules = {rule.key: rule for rule in RULES}
    selected_parts = []
    for _, choice in choices.iterrows():
        year = int(choice["year"])
        key = str(choice["rule"])
        source = formal if key == "BASE_EXIT" else adjust_exits(formal, raw, rules[key])
        selected_parts.append(source[source["test_year"].eq(year)].copy())
    selected = pd.concat(selected_parts, ignore_index=True, sort=False)
    holdout = selected[selected["test_year"].isin([2024, 2025, 2026])].copy()
    replay, summary = portfolio_metrics(holdout, raw, pool)
    replay["symbol"] = replay["symbol"].astype(str).str.zfill(6)

    price_map = raw.set_index(["symbol", "date"])["close"]
    audit_rows = []
    for _, trade in replay.iterrows():
        symbol = str(trade["symbol"]).zfill(6)
        entry_date = pd.Timestamp(trade["entry_date"])
        entry_raw = float(price_map.get((symbol, entry_date), np.nan))
        actual_exit = pd.to_datetime(trade.get("actual_exit_date"), errors="coerce")
        exit_raw = (
            float(price_map.get((symbol, actual_exit), np.nan))
            if pd.notna(actual_exit)
            else float(trade["actual_exit_close"])
        )
        audit_rows.append(
            {
                "symbol": symbol,
                "entry_date": entry_date.date().isoformat(),
                "entry_close": float(trade["entry_close"]),
                "entry_raw_close": entry_raw,
                "entry_error": float(trade["entry_close"]) - entry_raw,
                "actual_exit_date": actual_exit.date().isoformat() if pd.notna(actual_exit) else "",
                "actual_exit_close": float(trade["actual_exit_close"]),
                "exit_raw_close": exit_raw,
                "exit_error": float(trade["actual_exit_close"]) - exit_raw,
                "net_return": float(trade["net_return"]),
                "status": trade["status_portfolio"],
            }
        )
    audit = pd.DataFrame(audit_rows)
    expected_keys = expected[["symbol", "entry_date", "actual_exit_date", "net_return"]].copy()
    replay_keys = replay[["symbol", "entry_date", "actual_exit_date", "net_return"]].copy()
    for frame in (expected_keys, replay_keys):
        frame["entry_date"] = frame["entry_date"].astype(str)
        frame["actual_exit_date"] = frame["actual_exit_date"].fillna("").astype(str)
    compare = expected_keys.merge(
        replay_keys,
        on=["symbol", "entry_date", "actual_exit_date"],
        how="outer",
        suffixes=("_expected", "_replay"),
        indicator=True,
    )
    compare["return_error"] = compare["net_return_replay"] - compare["net_return_expected"]
    stale = holdout[holdout.get("stale_exit_triggered", False).fillna(False)].copy()
    stale_columns = [
        "test_year",
        "symbol",
        "entry_date",
        "entry_close",
        "stale_exit_rule",
        "stale_signal_date",
        "stale_signal_peak_return",
        "stale_signal_current_return",
        "exit_date",
        "exit_close",
    ]
    checks = {
        "annual_choices_use_past_only": bool((choices["train_end"] == choices["year"] - 1).all()),
        "replay_trade_keys_exact": bool(compare["_merge"].eq("both").all()),
        "replay_return_max_error": float(compare["return_error"].abs().max()),
        "entry_price_max_error": float(audit["entry_error"].abs().max()),
        "exit_price_max_error": float(audit["exit_error"].abs().max()),
        "stale_signals_execute_t_plus_1": True,
        "closed_trades": int(summary["closed_trades"]),
        "win_rate": float(summary["win_rate"]),
        "final_value": float(summary["final_value"]),
    }
    calendars = {
        symbol: pd.DatetimeIndex(group.sort_values("date")["date"])
        for symbol, group in raw.groupby("symbol")
    }
    for _, row in stale.iterrows():
        dates = calendars[str(row["symbol"])]
        signal_pos = int(dates.searchsorted(pd.Timestamp(row["stale_signal_date"])))
        if signal_pos + 1 >= len(dates) or dates[signal_pos + 1] != pd.Timestamp(row["exit_date"]):
            checks["stale_signals_execute_t_plus_1"] = False
    checks["passed"] = bool(
        checks["annual_choices_use_past_only"]
        and checks["replay_trade_keys_exact"]
        and checks["replay_return_max_error"] < 1e-12
        and checks["entry_price_max_error"] < 1e-12
        and checks["exit_price_max_error"] < 1e-12
        and checks["stale_signals_execute_t_plus_1"]
        and checks["win_rate"] == 1.0
        and checks["final_value"] > 18_000
    )
    audit.to_csv(OUT / "price_audit.csv", index=False, encoding="utf-8-sig")
    compare.to_csv(OUT / "replay_compare.csv", index=False, encoding="utf-8-sig")
    stale[stale_columns].to_csv(OUT / "stale_exit_events.csv", index=False, encoding="utf-8-sig")
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checks": checks,
    }
    (OUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
