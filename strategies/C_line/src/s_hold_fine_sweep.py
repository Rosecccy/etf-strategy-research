from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import controlled_upgrade_search as search
import take_profit_10_test as core
import walkforward_overlay_validation as wf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "s_hold_fine_sweep"


def build_frames(
    trades: pd.DataFrame, prices: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    frames = {"baseline": core.apply_take_profit(trades, prices, "baseline")}
    for arm in (0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115, 0.120):
        for floor in (0.060, 0.070, 0.075, 0.080, 0.085, 0.090, 0.095, 0.100):
            if floor >= arm:
                continue
            for hold in range(25, 36):
                rule = search.Rule("profit_floor", arm, floor, hold)
                frames[rule.key] = search.apply_rule(trades, prices, rule)
    return frames


def run_min_hold(
    frames: dict[str, pd.DataFrame],
    profiles: dict[str, dict[int, dict]],
    years: list[int],
    min_hold: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    allowed = {
        name: frame
        for name, frame in frames.items()
        if name == "baseline" or int(name.rsplit(":h", 1)[1]) >= min_hold
    }
    local_profiles = {name: profiles[name] for name in allowed}
    parts = []
    selections = []
    for year in years:
        rule, selection, _ = wf.choose_rule(local_profiles, year)
        part = frames[rule][pd.to_datetime(frames[rule]["entry_date"]).dt.year.eq(year)].copy()
        part["walkforward_rule"] = rule
        part["walkforward_test_year"] = year
        parts.append(part)
        selections.append(selection)
    combined = pd.concat(parts, ignore_index=True).sort_values(["entry_date", "symbol"])
    stats = {
        period: search.subset_stats(combined, period)
        for period in ("full", "dev", "holdout")
    }
    payload = {
        "min_hold": min_hold,
        "selected_years": int(sum(row["rule"] != "baseline" for row in selections)),
        **{
            f"{period}_{metric}": values[metric]
            for period, values in stats.items()
            for metric in ("final_value", "win_rate", "max_drawdown", "avg_annual_return")
        },
    }
    return payload, combined, pd.DataFrame(selections)


def main() -> None:
    core.configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = next(item for item in core.SPECS if item.key == "S")
    trades = core.load_trades(spec)
    prices = core.load_prices(spec.project, set(trades["symbol"]))
    frames = build_frames(trades, prices)
    years = sorted(pd.to_datetime(trades["entry_date"]).dt.year.unique())
    profiles = {name: wf.training_profile(frame, years) for name, frame in frames.items()}
    baseline = frames["baseline"]
    baseline_stats = {
        period: search.subset_stats(baseline, period)
        for period in ("full", "dev", "holdout")
    }

    rows = []
    generated = {}
    selected = {}
    for min_hold in range(27, 35):
        payload, combined, selections = run_min_hold(frames, profiles, years, min_hold)
        rows.append(payload)
        generated[min_hold] = combined
        selected[min_hold] = selections
    table = pd.DataFrame(rows)
    table["dev_ratio"] = table["dev_final_value"] / baseline_stats["dev"]["final_value"]
    table["holdout_ratio"] = table["holdout_final_value"] / baseline_stats["holdout"]["final_value"]
    table["full_ratio"] = table["full_final_value"] / baseline_stats["full"]["final_value"]
    table["passes"] = (
        table["dev_ratio"].gt(1.0)
        & table["holdout_ratio"].gt(1.0)
        & table["dev_win_rate"].ge(baseline_stats["dev"]["win_rate"] - 1e-12)
        & table["holdout_win_rate"].ge(baseline_stats["holdout"]["win_rate"] - 1e-12)
        & table["full_max_drawdown"].ge(baseline_stats["full"]["max_drawdown"] - 0.005)
    )
    passed = table[table["passes"]]
    winner = passed.iloc[(len(passed) - 1) // 2] if len(passed) else table.iloc[0]
    min_hold = int(winner["min_hold"])
    table.to_csv(OUT / "sweep.csv", index=False, encoding="utf-8-sig")
    generated[min_hold].to_csv(OUT / "winner_trades.csv", index=False, encoding="utf-8-sig")
    selected[min_hold].to_csv(OUT / "winner_selections.csv", index=False, encoding="utf-8-sig")
    summary = {
        "method": "annual expanding-window minimum-hold daily fine sweep",
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
