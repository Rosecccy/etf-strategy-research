from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import fear_greed_oos as fg
import fear_greed_weight_sweep as sweep
from control_sizing_test import account


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "regime_threshold_test"
DEV_YEARS = (2021, 2022, 2023)
HOLDOUT_YEARS = (2025, 2026)
QVIX_WEIGHT = 0.20
GREED_EXIT = 70.0
MAX_HOLD = 90
FEAR_THRESHOLDS = (45, 55, 60, 65, 70)
NEUTRAL_THRESHOLDS = (40, 45, 50, 55, 60)
MIN_HOLDOUT_CLOSED_TRADES = 5


def selection_score(summary: dict[str, float | int]) -> float:
    return (
        math.log(max(float(summary["final_value"]), 1.0) / fg.INITIAL_CAPITAL)
        + 0.25 * float(np.nan_to_num(summary["win_rate"], nan=0.0))
        + 0.70 * float(summary["max_drawdown"])
    )


def threshold_mask(
    frame: pd.DataFrame,
    fear_threshold: int,
    neutral_threshold: int,
) -> pd.Series:
    required = np.select(
        [
            frame["regime"].eq("恐惧"),
            frame["regime"].eq("中性"),
        ],
        [fear_threshold, neutral_threshold],
        default=45,
    )
    return frame["fear_weighted"].ge(required)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, adjusted_map, raw, pool, _ = sweep.generate()
    formal = adjusted_map[(QVIX_WEIGHT, MAX_HOLD, GREED_EXIT)].copy()
    formal["overlay"] = "fear_greed"

    variants: dict[str, pd.DataFrame] = {
        "baseline_f45_n45": formal[formal["fear_weighted"].ge(45)].copy()
    }
    for fear in FEAR_THRESHOLDS:
        for neutral in NEUTRAL_THRESHOLDS:
            key = f"fear_{fear}_neutral_{neutral}"
            variants[key] = formal[
                threshold_mask(formal, fear, neutral)
            ].copy()

    rows: list[dict[str, object]] = []
    logs: dict[tuple[str, str], pd.DataFrame] = {}
    summaries: dict[tuple[str, str], dict[str, float | int]] = {}
    for key, source in variants.items():
        for stage, years in (
            ("development", DEV_YEARS),
            ("holdout", HOLDOUT_YEARS),
            ("all", None),
        ):
            local = (
                source
                if years is None
                else source[source["test_year"].isin(years)].copy()
            )
            log, summary = account(local, raw, pool, "full")
            logs[(key, stage)] = log
            summaries[(key, stage)] = summary
            rows.append(
                {
                    "variant": key,
                    "stage": stage,
                    "selection_score": selection_score(summary),
                    **summary,
                }
            )
        print(f"[D regime threshold] {key}")
    result = pd.DataFrame(rows)
    development = result[result["stage"].eq("development")].sort_values(
        ["selection_score", "final_value"], ascending=False
    )
    winner = str(development.iloc[0]["variant"])
    baseline = summaries[("baseline_f45_n45", "holdout")]
    winner_holdout = summaries[(winner, "holdout")]

    parts = winner.replace("fear_", "").replace("neutral_", "").split("_")
    winner_fear = int(parts[0]) if winner != "baseline_f45_n45" else 45
    winner_neutral = int(parts[1]) if winner != "baseline_f45_n45" else 45
    neighbor_keys = [
        f"fear_{fear}_neutral_{neutral}"
        for fear in FEAR_THRESHOLDS
        for neutral in NEUTRAL_THRESHOLDS
        if abs(fear - winner_fear) <= 5
        and abs(neutral - winner_neutral) <= 5
    ]
    neighbor_passes = int(
        sum(
            summaries[(key, "holdout")]["final_value"] >= baseline["final_value"]
            and summaries[(key, "holdout")]["win_rate"]
            >= baseline["win_rate"] - 0.02
            and summaries[(key, "holdout")]["max_drawdown"]
            >= baseline["max_drawdown"] - 0.01
            for key in neighbor_keys
        )
    )
    promoted = bool(
        winner != "baseline_f45_n45"
        and int(winner_holdout["closed_trades"])
        >= MIN_HOLDOUT_CLOSED_TRADES
        and winner_holdout["final_value"] > baseline["final_value"]
        and winner_holdout["win_rate"] >= baseline["win_rate"]
        and winner_holdout["max_drawdown"] >= baseline["max_drawdown"]
        and neighbor_passes >= 3
    )

    result.to_csv(OUT / "variants.csv", index=False, encoding="utf-8-sig")
    logs[(winner, "all")].to_csv(
        OUT / "winner_trades.csv", index=False, encoding="utf-8-sig"
    )
    logs[(winner, "holdout")].to_csv(
        OUT / "winner_holdout_trades.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "control_variable": "fear_entry_threshold_by_market_regime_only",
        "fixed_rules": {
            "qvix_weight": QVIX_WEIGHT,
            "greed_exit": GREED_EXIT,
            "max_hold": MAX_HOLD,
            "execution": "T+1 close",
            "portfolio": "one position, full allocation, 100-share lots",
            "fee": "0.03%, minimum CNY 5 per side",
        },
        "development_selected_variant": winner,
        "baseline_holdout": baseline,
        "winner_holdout": winner_holdout,
        "minimum_holdout_closed_trades_for_promotion": (
            MIN_HOLDOUT_CLOSED_TRADES
        ),
        "nearby_variants": neighbor_keys,
        "nearby_variants_passing_holdout_floor": neighbor_passes,
        "promoted": promoted,
        "formal_config_changed": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
