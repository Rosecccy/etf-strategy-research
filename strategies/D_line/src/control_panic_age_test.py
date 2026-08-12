from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import fear_greed_oos as fg
import fear_greed_weight_sweep as sweep


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "panic_age_test"
DEV_YEARS = (2021, 2022, 2023)
HOLDOUT_YEARS = (2025, 2026)
AGES = (1, 2, 3, 4, 5, 7, 10)
ENTRY_THRESHOLD = 45.0


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def add_panic_age(
    sentiment: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    daily = sentiment[["date", "fear_core", "fear_qvix"]].copy()
    daily["fear_weighted"] = sweep.composite(
        daily["fear_core"], daily["fear_qvix"], 0.2
    )
    active = daily["fear_weighted"].ge(ENTRY_THRESHOLD).fillna(False)
    block = active.ne(active.shift()).cumsum()
    daily["panic_age"] = active.groupby(block).cumcount() + 1
    daily.loc[~active, "panic_age"] = 0
    return candidates.merge(
        daily[["date", "panic_age"]].rename(columns={"date": "signal_date"}),
        on="signal_date",
        how="left",
    )


def portfolio_metrics(
    candidates: pd.DataFrame,
    raw: pd.DataFrame,
    pool: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    log, summary = fg.portfolio_backtest(candidates, raw, pool, 1, "fear")
    realized = (
        log[log["status_portfolio"].eq("closed")].sort_values("actual_exit_date")
        if len(log)
        else log
    )
    marked = (
        log[log["status_portfolio"].eq("open_marked")]
        if len(log)
        else log
    )
    sequence = pd.concat([realized, marked], ignore_index=True)
    if len(sequence):
        equity = (1.0 + sequence["net_return"].astype(float)).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        summary["max_drawdown"] = float(drawdown.min())
    else:
        summary["max_drawdown"] = 0.0
    return log, summary


def score(stat: dict) -> float:
    return (
        np.log(max(float(stat["final_value"]), 1.0) / 10_000.0)
        + 0.30 * float(np.nan_to_num(stat["win_rate"], nan=0.0))
        + 0.70 * float(stat["max_drawdown"])
    )


def main() -> None:
    configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    table, sentiment, adjusted, raw, pool, _ = sweep.generate()
    winner = sweep.stable(table).iloc[0]
    _, formal = sweep.evaluate_years(winner, adjusted)
    formal = add_panic_age(sentiment, formal)

    rows: list[dict] = []
    logs: dict[tuple[int, str], pd.DataFrame] = {}
    summaries: dict[tuple[int, str], dict] = {}
    for age in AGES:
        selected = formal[formal["panic_age"].ge(age)].copy()
        for stage, years in (
            ("development", DEV_YEARS),
            ("holdout", HOLDOUT_YEARS),
        ):
            local = selected[selected["test_year"].isin(years)].copy()
            log, stat = portfolio_metrics(local, raw, pool)
            logs[(age, stage)] = log
            summaries[(age, stage)] = stat
            rows.append(
                {
                    "min_panic_age": age,
                    "stage": stage,
                    "score": score(stat),
                    **stat,
                }
            )

    result = pd.DataFrame(rows)
    dev = result[result["stage"].eq("development")].sort_values(
        ["score", "final_value", "win_rate"], ascending=False
    )
    chosen = int(dev.iloc[0]["min_panic_age"])
    baseline = result[
        result["stage"].eq("holdout") & result["min_panic_age"].eq(1)
    ].iloc[0]
    held = result[
        result["stage"].eq("holdout") & result["min_panic_age"].eq(chosen)
    ].iloc[0]
    neighbors = [
        age for age in AGES if abs(age - chosen) <= 2
    ]
    passing = result[
        result["stage"].eq("holdout")
        & result["min_panic_age"].isin(neighbors)
        & result["final_value"].gt(float(baseline["final_value"]) * 1.03)
        & result["win_rate"].ge(float(baseline["win_rate"]))
        & result["max_drawdown"].ge(float(baseline["max_drawdown"]))
        & result["closed_trades"].ge(5)
    ]
    promoted = bool(
        chosen != 1
        and held["final_value"] > baseline["final_value"] * 1.03
        and held["win_rate"] >= baseline["win_rate"]
        and held["max_drawdown"] >= baseline["max_drawdown"]
        and held["closed_trades"] >= 5
        and len(passing) >= 2
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "control_variable": "consecutive_panic_days_before_entry_only",
        "fixed_rules": {
            "qvix_weight": 0.2,
            "entry_threshold": 45,
            "greed_exit": 70,
            "max_hold": 90,
            "execution": "T+1 close",
            "portfolio": "one position, full allocation",
        },
        "development_selected_age": chosen,
        "baseline_holdout": baseline.to_dict(),
        "winner_holdout": held.to_dict(),
        "nearby_variants_passing": int(len(passing)),
        "promoted": promoted,
    }
    result.to_csv(OUT / "variants.csv", index=False, encoding="utf-8-sig")
    logs[(chosen, "holdout")].to_csv(
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
