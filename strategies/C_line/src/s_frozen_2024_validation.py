from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import controlled_upgrade_search as search
import s_robust_plateau_sweep as plateau
import take_profit_10_test as core
import walkforward_overlay_validation as wf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "s_frozen_2024"
FREEZE_YEAR = 2024


def main() -> None:
    core.configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = next(item for item in core.SPECS if item.key == "S")
    trades = core.load_trades(spec)
    prices = core.load_prices(spec.project, set(trades["symbol"]))
    frames = wf.candidate_frames(spec, trades, prices)
    years = sorted(pd.to_datetime(trades["entry_date"]).dt.year.unique())
    profiles = {name: wf.training_profile(frame, years) for name, frame in frames.items()}
    baseline = frames["baseline"]
    baseline_stats = {
        period: search.subset_stats(baseline, period)
        for period in ("full", "dev", "holdout")
    }
    entry_years = pd.to_datetime(baseline["entry_date"]).dt.year
    rows = []
    generated = {}
    for fraction in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        rule, selection = plateau.choose_plateau_rule(profiles, FREEZE_YEAR, fraction)
        candidate = frames[rule]
        candidate_years = pd.to_datetime(candidate["entry_date"]).dt.year
        combined = pd.concat(
            [baseline[entry_years.lt(FREEZE_YEAR)], candidate[candidate_years.ge(FREEZE_YEAR)]],
            ignore_index=True,
        ).sort_values(["entry_date", "symbol"])
        stats = {
            period: search.subset_stats(combined, period)
            for period in ("full", "dev", "holdout")
        }
        row = {
            "plateau_fraction": fraction,
            "rule": rule,
            "plateau_size": selection["plateau_size"],
            **{
                f"{period}_{metric}": values[metric]
                for period, values in stats.items()
                for metric in ("final_value", "win_rate", "max_drawdown", "avg_annual_return")
            },
        }
        rows.append(row)
        generated[fraction] = combined
    table = pd.DataFrame(rows)
    table["holdout_ratio"] = table["holdout_final_value"] / baseline_stats["holdout"]["final_value"]
    table["full_ratio"] = table["full_final_value"] / baseline_stats["full"]["final_value"]
    table["passes"] = (
        table["holdout_ratio"].gt(1.0)
        & table["holdout_win_rate"].ge(baseline_stats["holdout"]["win_rate"] - 1e-12)
        & table["holdout_max_drawdown"].ge(baseline_stats["holdout"]["max_drawdown"] - 0.005)
    )
    passed = table[table["passes"]]
    winner = passed.iloc[(len(passed) - 1) // 2] if len(passed) else table.iloc[0]
    fraction = float(winner["plateau_fraction"])
    table.to_csv(OUT / "sweep.csv", index=False, encoding="utf-8-sig")
    generated[fraction].to_csv(OUT / "winner_trades.csv", index=False, encoding="utf-8-sig")
    summary = {
        "method": "select with data through 2023, freeze from 2024 onward",
        "baseline": baseline_stats,
        "passing_neighbor_count": int(len(passed)),
        "winner": winner.to_dict(),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
