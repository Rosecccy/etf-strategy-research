from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import fear_greed_oos as fg
import fear_greed_weight_sweep as sweep
from control_panic_age_test import portfolio_metrics, score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "exit_portfolio_test"
DEV_YEARS = (2021, 2022, 2023)
HOLDOUT_YEARS = (2025, 2026)
HOLDS = (60, 90)
GREEDS = (65.0, 70.0, 75.0)


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main() -> None:
    configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    train_grid, _, adjusted, raw, pool, _ = sweep.generate()
    formal_winner = sweep.stable(train_grid).iloc[0]
    threshold = float(formal_winner["entry_threshold"])
    rows: list[dict] = []
    logs: dict[tuple[int, float, str], pd.DataFrame] = {}

    for hold in HOLDS:
        for greed in GREEDS:
            candidates = adjusted[(0.2, hold, greed)]
            candidates = candidates[candidates["fear_weighted"].ge(threshold)].copy()
            for stage, years in (
                ("development", DEV_YEARS),
                ("holdout", HOLDOUT_YEARS),
            ):
                local = candidates[candidates["test_year"].isin(years)].copy()
                log, stat = portfolio_metrics(local, raw, pool)
                logs[(hold, greed, stage)] = log
                rows.append(
                    {
                        "max_hold": hold,
                        "greed_exit": greed,
                        "stage": stage,
                        "score": score(stat),
                        **stat,
                    }
                )

    result = pd.DataFrame(rows)
    dev = result[result["stage"].eq("development")].sort_values(
        ["score", "final_value", "win_rate"], ascending=False
    )
    chosen_hold = int(dev.iloc[0]["max_hold"])
    chosen_greed = float(dev.iloc[0]["greed_exit"])
    baseline = result[
        result["stage"].eq("holdout")
        & result["max_hold"].eq(90)
        & result["greed_exit"].eq(70)
    ].iloc[0]
    held = result[
        result["stage"].eq("holdout")
        & result["max_hold"].eq(chosen_hold)
        & result["greed_exit"].eq(chosen_greed)
    ].iloc[0]
    nearby = result[
        result["stage"].eq("holdout")
        & result["max_hold"].isin(HOLDS)
        & result["greed_exit"].between(chosen_greed - 5, chosen_greed + 5)
    ]
    passing = nearby[
        nearby["final_value"].gt(float(baseline["final_value"]) * 1.03)
        & nearby["win_rate"].ge(float(baseline["win_rate"]) - 0.05)
        & nearby["max_drawdown"].ge(float(baseline["max_drawdown"]) - 0.01)
        & nearby["closed_trades"].ge(5)
    ]
    promoted = bool(
        (chosen_hold, chosen_greed) != (90, 70.0)
        and held["final_value"] > baseline["final_value"] * 1.03
        and held["win_rate"] >= baseline["win_rate"] - 0.05
        and held["max_drawdown"] >= baseline["max_drawdown"] - 0.01
        and held["closed_trades"] >= 5
        and len(passing) >= 2
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "control_variable": "portfolio_level_exit_timing_only",
        "fixed_rules": {
            "qvix_weight": 0.2,
            "entry_threshold": threshold,
            "execution": "T+1 close",
            "portfolio": "one position, full allocation",
        },
        "development_selected": {
            "max_hold": chosen_hold,
            "greed_exit": chosen_greed,
        },
        "baseline_holdout": baseline.to_dict(),
        "winner_holdout": held.to_dict(),
        "nearby_variants_passing": int(len(passing)),
        "promoted": promoted,
    }
    result.to_csv(OUT / "variants.csv", index=False, encoding="utf-8-sig")
    logs[(chosen_hold, chosen_greed, "holdout")].to_csv(
        OUT / "winner_holdout_trades.csv", index=False, encoding="utf-8-sig"
    )
    passing.to_csv(OUT / "passing_neighbors.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
