from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from exit_overlay_test import metrics, net_account, score
from hold_length_test import load_trades, shift_fallback_exits


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "rolling_hold_selector_test"
DEV = set(range(2019, 2024))
HOLDOUT = {2024, 2025, 2026}
SHIFTS = [-20, -15, -10, -7, -5, -3, 0, 3, 5, 7, 10, 15, 20, 30, 40]


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def training_years(year: int, window: int | None) -> set[int]:
    first = 2014 if window is None else max(2014, year - window)
    return set(range(first, year))


def rolling_account(
    accounts: dict[int, pd.DataFrame],
    window: int | None,
    max_abs_shift: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual_rows: list[pd.DataFrame] = []
    choices: list[dict] = []
    available = [shift for shift in SHIFTS if abs(shift) <= max_abs_shift]
    for year in range(2016, 2027):
        train = training_years(year, window)
        ranked: list[tuple[float, float, float, int]] = []
        for shift in available:
            stat = metrics(accounts[shift], train)
            ranked.append(
                (
                    score(stat),
                    float(stat["final_1000"]),
                    -abs(shift),
                    shift,
                )
            )
        ranked.sort(reverse=True)
        chosen = int(ranked[0][3])
        year_rows = accounts[chosen][
            accounts[chosen]["entry_date"].dt.year.eq(year)
        ].copy()
        year_rows["selected_hold_shift"] = chosen
        year_rows["selector_window"] = "all" if window is None else window
        annual_rows.append(year_rows)
        choices.append(
            {
                "year": year,
                "train_start": min(train),
                "train_end": max(train),
                "window": "all" if window is None else window,
                "max_abs_shift": max_abs_shift,
                "chosen_shift": chosen,
                "train_score": ranked[0][0],
            }
        )
    return (
        pd.concat(annual_rows, ignore_index=True).sort_values("entry_date"),
        pd.DataFrame(choices),
    )


def main() -> None:
    configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = load_trades()
    accounts = {
        shift: baseline.copy() if shift == 0 else shift_fallback_exits(baseline, shift)
        for shift in SHIFTS
    }
    base_dev = metrics(baseline, DEV)
    base_holdout = metrics(baseline, HOLDOUT)
    rows: list[dict] = []
    generated: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for window in (3, 5, None):
        for cap in (10, 20, 40):
            rule_id = f"w{'all' if window is None else window}_cap{cap}"
            account, choices = rolling_account(accounts, window, cap)
            generated[rule_id] = (account, choices)
            dev = metrics(account, DEV)
            holdout = metrics(account, HOLDOUT)
            rows.append(
                {
                    "rule_id": rule_id,
                    "window": "all" if window is None else window,
                    "max_abs_shift": cap,
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
    winner_id = str(winner["rule_id"])
    account, choices = generated[winner_id]
    passing = table[
        table["dev_final_1000"].ge(float(winner["dev_final_1000"]) * 0.95)
        & table["holdout_final_1000"].gt(base_holdout["final_1000"] * 1.03)
        & table["holdout_win_rate"].ge(base_holdout["win_rate"])
        & table["holdout_max_drawdown"].ge(base_holdout["max_drawdown"])
    ]
    promoted = bool(
        winner["holdout_final_1000"] > base_holdout["final_1000"] * 1.03
        and winner["holdout_win_rate"] >= base_holdout["win_rate"]
        and winner["holdout_max_drawdown"] >= base_holdout["max_drawdown"]
        and len(passing) >= 2
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test": "S annual rolling hold-shift selector",
        "baseline_dev": base_dev,
        "baseline_holdout": base_holdout,
        "development_winner": winner.to_dict(),
        "passing_neighbors": int(len(passing)),
        "baseline_net_10000": net_account(baseline),
        "winner_net_10000": net_account(account),
        "promoted": promoted,
    }
    table.to_csv(OUT / "variants.csv", index=False, encoding="utf-8-sig")
    account.to_csv(OUT / "winner_trades.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(OUT / "winner_choices.csv", index=False, encoding="utf-8-sig")
    passing.to_csv(OUT / "passing_neighbors.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
