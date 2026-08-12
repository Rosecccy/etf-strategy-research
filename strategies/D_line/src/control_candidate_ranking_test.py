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
OUT = ROOT / "out" / "candidate_ranking_test"
DEV_YEARS = (2021, 2022, 2023)
HOLDOUT_YEARS = (2025, 2026)
MODES = (
    "current_quality",
    "symbol_fear",
    "liquidity",
    "historical",
    "symbol_history_70_30",
    "symbol_history_50_50",
    "symbol_history_30_70",
)


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def add_causal_history(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(["entry_date", "symbol"]).copy()
    hist_win: list[float] = []
    hist_avg: list[float] = []
    hist_n: list[int] = []
    for row in result.itertuples():
        prior = result[
            result["symbol"].eq(row.symbol)
            & result["status"].eq("closed")
            & result["exit_date"].lt(row.entry_date)
            & result["return_rate"].notna()
        ]["return_rate"].astype(float)
        count = int(len(prior))
        hist_n.append(count)
        # Shrink sparse symbol histories toward a neutral 50% / 0% prior.
        hist_win.append(float(((prior > 0).sum() + 2.0) / (count + 4.0)))
        hist_avg.append(float(prior.sum() / (count + 5.0)))
    result["hist_n"] = hist_n
    result["hist_win"] = hist_win
    result["hist_avg"] = hist_avg
    result["historical_score"] = (
        70.0 * result["hist_win"] + 30.0 * (0.5 + result["hist_avg"] * 5.0)
    )
    return result


def ranked_candidates(frame: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, str]:
    local = frame.copy()
    if mode == "liquidity":
        return local, "liquidity"
    if mode == "current_quality":
        return local, "fear"
    if mode == "symbol_fear":
        local["buy_quality"] = local["symbol_fear"]
    elif mode == "historical":
        local["buy_quality"] = local["historical_score"]
    else:
        symbol_weight = float(mode.split("_")[-2]) / 100.0
        local["buy_quality"] = (
            symbol_weight * local["symbol_fear"]
            + (1.0 - symbol_weight) * local["historical_score"]
        )
    return local, "fear"


def main() -> None:
    configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    table, _, adjusted, raw, pool, _ = sweep.generate()
    winner = sweep.stable(table).iloc[0]
    _, formal = sweep.evaluate_years(winner, adjusted)
    formal = add_causal_history(formal)
    rows: list[dict] = []
    logs: dict[tuple[str, str], pd.DataFrame] = {}

    for mode in MODES:
        candidates, ranking = ranked_candidates(formal, mode)
        for stage, years in (
            ("development", DEV_YEARS),
            ("holdout", HOLDOUT_YEARS),
        ):
            local = candidates[candidates["test_year"].isin(years)].copy()
            log, stat = portfolio_metrics(local, raw, pool)
            logs[(mode, stage)] = log
            rows.append(
                {
                    "ranking_mode": mode,
                    "stage": stage,
                    "score": score(stat),
                    **stat,
                }
            )

    result = pd.DataFrame(rows)
    dev = result[result["stage"].eq("development")].sort_values(
        ["score", "final_value", "win_rate"], ascending=False
    )
    chosen = str(dev.iloc[0]["ranking_mode"])
    baseline = result[
        result["stage"].eq("holdout")
        & result["ranking_mode"].eq("current_quality")
    ].iloc[0]
    held = result[
        result["stage"].eq("holdout") & result["ranking_mode"].eq(chosen)
    ].iloc[0]
    family = result[
        result["stage"].eq("holdout")
        & result["ranking_mode"].str.startswith("symbol_history")
    ]
    passing = family[
        family["final_value"].gt(float(baseline["final_value"]) * 1.03)
        & family["win_rate"].ge(float(baseline["win_rate"]))
        & family["max_drawdown"].ge(float(baseline["max_drawdown"]))
        & family["closed_trades"].ge(3)
    ]
    promoted = bool(
        chosen != "current_quality"
        and held["final_value"] > baseline["final_value"] * 1.03
        and held["win_rate"] >= baseline["win_rate"]
        and held["max_drawdown"] >= baseline["max_drawdown"]
        and held["closed_trades"] >= 3
        and (
            not chosen.startswith("symbol_history")
            or len(passing) >= 2
        )
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "control_variable": "same_day_candidate_ranking_only",
        "fixed_rules": {
            "qvix_weight": 0.2,
            "entry_threshold": 45,
            "greed_exit": 70,
            "max_hold": 90,
            "execution": "T+1 close",
            "portfolio": "one position, full allocation",
        },
        "development_selected_ranking": chosen,
        "baseline_holdout": baseline.to_dict(),
        "winner_holdout": held.to_dict(),
        "blend_neighbors_passing": int(len(passing)),
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
