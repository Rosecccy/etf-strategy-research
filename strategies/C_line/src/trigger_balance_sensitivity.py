from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import controlled_upgrade_search as stats_mod
import cross_line_gap_overlay as gap
import take_profit_10_test as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "trigger_balanced_gap"


def run_config(
    target: pd.DataFrame,
    candidates: dict[tuple[str, ...], pd.DataFrame],
    win_weight: float,
    trigger_weight: float,
    dd_weight: float,
) -> tuple[pd.DataFrame, list[dict]]:
    years = sorted(pd.to_datetime(target["entry_date"]).dt.year.unique())
    parts = []
    selections = []
    for year in years:
        base_train = target[pd.to_datetime(target["entry_date"]).dt.year.lt(year)].copy()
        winner_priority = None
        best = None
        if len(base_train) >= 12:
            base = core.simulate_account(base_train)[0]
            for priority, schedule in candidates.items():
                train = schedule[pd.to_datetime(schedule["entry_date"]).dt.year.lt(year)].copy()
                current = core.simulate_account(train)[0]
                added = int(train["is_gap_overlay"].fillna(False).sum())
                trigger_ratio = len(train) / len(base_train)
                final_ratio = current["final_value"] / base["final_value"]
                win_delta = current["win_rate"] - base["win_rate"]
                dd_delta = current["max_drawdown"] - base["max_drawdown"]
                score = (
                    np.log(max(final_ratio, 1e-12))
                    + win_weight * win_delta
                    + trigger_weight * np.log(max(trigger_ratio, 1e-12))
                    + dd_weight * min(dd_delta, 0.10)
                )
                passes = bool(
                    added >= 2
                    and trigger_ratio >= 1.10
                    and final_ratio >= 0.995
                    and win_delta >= -0.02
                    and dd_delta >= -0.01
                    and score > 0.0
                )
                if passes and (best is None or score > best["score"]):
                    winner_priority = priority
                    best = {
                        "test_year": int(year),
                        "priority": ">".join(priority),
                        "score": score,
                    }
        source = target if winner_priority is None else candidates[winner_priority]
        local = source[pd.to_datetime(source["entry_date"]).dt.year.eq(year)].copy()
        parts.append(local)
        selections.append(best or {"test_year": int(year), "priority": "baseline", "score": 0.0})
    return pd.concat(parts, ignore_index=True).sort_values(["entry_date", "is_gap_overlay", "symbol"]), selections


def main() -> None:
    core.configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {line: gap.load_frame(line) for line in ("C", "S", "D")}
    prices = gap.load_all_prices(frames)
    target = frames["D"]
    candidates = {
        priority: gap.build_schedule("D", target, frames, priority, prices)
        for priority in gap.overlay_options("D")
    }
    baseline = {period: stats_mod.subset_stats(target, period) for period in ("full", "holdout")}
    rows = []
    for win_weight, trigger_weight, dd_weight in itertools.product(
        (0.30, 0.50, 0.70),
        (0.05, 0.10, 0.15),
        (0.10, 0.20, 0.30),
    ):
        winner, selections = run_config(
            target, candidates, win_weight, trigger_weight, dd_weight
        )
        current = {period: stats_mod.subset_stats(winner, period) for period in ("full", "holdout")}
        rows.append(
            {
                "win_weight": win_weight,
                "trigger_weight": trigger_weight,
                "dd_weight": dd_weight,
                "trades": len(winner),
                "added": int(winner["is_gap_overlay"].fillna(False).sum()),
                "full_ratio": current["full"]["final_value"] / baseline["full"]["final_value"],
                "full_win": current["full"]["win_rate"],
                "avg_annual": current["full"]["avg_annual_return"],
                "max_drawdown": current["full"]["max_drawdown"],
                "holdout_ratio": current["holdout"]["final_value"] / baseline["holdout"]["final_value"],
                "holdout_win": current["holdout"]["win_rate"],
                "selected_2024": next(
                    (item["priority"] for item in selections if item["test_year"] == 2024), "missing"
                ),
            }
        )
    table = pd.DataFrame(rows)
    table["strict_improvement"] = (
        table["full_ratio"].gt(1.0)
        & table["holdout_ratio"].gt(1.0)
        & table["full_win"].ge(baseline["full"]["win_rate"])
        & table["holdout_win"].ge(baseline["holdout"]["win_rate"])
        & table["trades"].ge(len(target))
    )
    table.to_csv(OUT / "sensitivity.csv", index=False, encoding="utf-8-sig")
    strict = table[table["strict_improvement"]]
    payload = {
        "grid_size": len(table),
        "strict_improvements": len(strict),
        "strict_rate": len(strict) / len(table),
        "median_strict_full_ratio": None if strict.empty else float(strict["full_ratio"].median()),
        "median_strict_holdout_ratio": None if strict.empty else float(strict["holdout_ratio"].median()),
        "selected_2024_counts": table["selected_2024"].value_counts().to_dict(),
    }
    (OUT / "sensitivity.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
