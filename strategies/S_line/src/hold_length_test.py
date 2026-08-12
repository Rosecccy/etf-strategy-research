from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from exit_overlay_test import (
    DEV,
    HOLDOUT,
    load_trades,
    metrics,
    net_account,
    next_main_after,
    next_main_dates,
    price_frame,
    score,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "hold_length_test"


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def shift_fallback_exits(frame: pd.DataFrame, shift: int) -> pd.DataFrame:
    main_entries = next_main_dates(frame)
    rows: list[dict] = []
    for _, trade in frame.iterrows():
        item = trade.to_dict()
        if str(trade["source"]) == "主策略":
            item["hold_shift"] = 0
            rows.append(item)
            continue

        prices = price_frame(str(trade["symbol"]))
        dates = pd.DatetimeIndex(prices["date"])
        entry_pos = int(dates.searchsorted(trade["entry_date"], side="left"))
        exit_pos = int(dates.searchsorted(trade["exit_date"], side="left"))
        new_pos = max(entry_pos + 7, min(exit_pos + shift, len(prices) - 1))
        next_main = next_main_after(main_entries, pd.Timestamp(trade["entry_date"]))
        if next_main is not None:
            main_pos = int(dates.searchsorted(next_main, side="left"))
            new_pos = min(new_pos, main_pos)

        entry_price = float(prices.iloc[entry_pos]["close"])
        exit_price = float(prices.iloc[new_pos]["close"])
        item["exit_date"] = pd.Timestamp(prices.iloc[new_pos]["date"])
        item["entry_close"] = entry_price
        item["exit_close"] = exit_price
        item["ret"] = exit_price / entry_price - 1.0
        item["hold_shift"] = shift
        rows.append(item)

    proposed = pd.DataFrame(rows).sort_values(["entry_date", "symbol"])
    kept: list[pd.Series] = []
    busy_until = pd.Timestamp.min
    for _, row in proposed.iterrows():
        if pd.Timestamp(row["entry_date"]) < busy_until:
            continue
        kept.append(row)
        busy_until = pd.Timestamp(row["exit_date"])
    return pd.DataFrame(kept).reset_index(drop=True)


def main() -> None:
    configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = load_trades()
    shifts = [-20, -15, -10, -7, -5, -3, 0, 3, 5, 7, 10, 15, 20, 30, 40]
    rows: list[dict] = []
    accounts: dict[int, pd.DataFrame] = {}

    for shift in shifts:
        account = baseline.copy() if shift == 0 else shift_fallback_exits(baseline, shift)
        accounts[shift] = account
        dev = metrics(account, DEV)
        holdout = metrics(account, HOLDOUT)
        rows.append(
            {
                "shift_days": shift,
                "dev_score": score(dev),
                **{f"dev_{key}": value for key, value in dev.items()},
                **{f"holdout_{key}": value for key, value in holdout.items()},
            }
        )

    table = pd.DataFrame(rows).sort_values(
        ["dev_score", "dev_final_1000", "dev_win_rate"],
        ascending=False,
    )
    winner = table.iloc[0]
    winner_shift = int(winner["shift_days"])
    base = table.loc[table["shift_days"].eq(0)].iloc[0]
    nearby = table[
        table["shift_days"].between(winner_shift - 5, winner_shift + 5)
        & table["dev_final_1000"].ge(float(winner["dev_final_1000"]) * 0.95)
    ]
    passing = nearby[
        nearby["holdout_final_1000"].gt(float(base["holdout_final_1000"]) * 1.03)
        & nearby["holdout_win_rate"].ge(float(base["holdout_win_rate"]))
        & nearby["holdout_max_drawdown"].ge(float(base["holdout_max_drawdown"]))
    ]
    promoted = bool(
        winner_shift != 0
        and winner["holdout_final_1000"] > base["holdout_final_1000"] * 1.03
        and winner["holdout_win_rate"] >= base["holdout_win_rate"]
        and winner["holdout_max_drawdown"] >= base["holdout_max_drawdown"]
        and len(passing) >= 2
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test": "S fallback fixed hold-length shift",
        "development_years": "2016-2023",
        "holdout_years": "2024-2026",
        "baseline": base.to_dict(),
        "development_winner": winner.to_dict(),
        "nearby_development_rules": int(len(nearby)),
        "passing_neighbors": int(len(passing)),
        "baseline_net_10000": net_account(baseline),
        "winner_net_10000": net_account(accounts[winner_shift]),
        "promoted": promoted,
    }
    table.to_csv(OUT / "variants.csv", index=False, encoding="utf-8-sig")
    accounts[winner_shift].to_csv(
        OUT / "winner_trades.csv", index=False, encoding="utf-8-sig"
    )
    passing.to_csv(OUT / "passing_neighbors.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
