from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import controlled_upgrade_search as search
import take_profit_10_test as core
import walkforward_overlay_validation as wf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "d_robust_plateau"
MIN_TRAIN_TRADES = 12
MIN_TRAIN_TRIGGERS = 2


def target_of(rule: str) -> float:
    return float(rule.split(":a", 1)[1].split(":", 1)[0])


def choose_plateau_rule(
    profiles: dict[str, dict[int, dict]], year: int, fraction: float
) -> tuple[str, dict]:
    baseline = profiles["baseline"][year]
    candidates = []
    for name, profile in profiles.items():
        if name == "baseline":
            continue
        current = profile[year]
        ratio = current["final_value"] / baseline["final_value"]
        win_delta = current["win_rate"] - baseline["win_rate"]
        dd_delta = current["max_drawdown"] - baseline["max_drawdown"]
        if (
            current["trades"] >= MIN_TRAIN_TRADES
            and current["trigger_count"] >= MIN_TRAIN_TRIGGERS
            and ratio > 1.0
            and win_delta >= -1e-12
            and dd_delta >= -0.005
        ):
            candidates.append(
                {
                    "rule": name,
                    "target": target_of(name),
                    "ratio": ratio,
                    "triggers": current["trigger_count"],
                }
            )
    if not candidates:
        return "baseline", {
            "year": year,
            "rule": "baseline",
            "target": None,
            "plateau_size": 0,
            "best_ratio": 1.0,
            "cutoff_ratio": 1.0,
        }
    best = max(item["ratio"] for item in candidates)
    cutoff = 1.0 + fraction * (best - 1.0)
    plateau = sorted(
        (item for item in candidates if item["ratio"] >= cutoff),
        key=lambda item: item["target"],
    )
    winner = plateau[(len(plateau) - 1) // 2]
    return str(winner["rule"]), {
        "year": year,
        "rule": winner["rule"],
        "target": winner["target"],
        "plateau_size": len(plateau),
        "best_ratio": best,
        "cutoff_ratio": cutoff,
    }


def run_fraction(
    frames: dict[str, pd.DataFrame],
    profiles: dict[str, dict[int, dict]],
    years: list[int],
    fraction: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    parts = []
    selections = []
    for year in years:
        rule, selection = choose_plateau_rule(profiles, year, fraction)
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
        "plateau_fraction": fraction,
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
    spec = next(item for item in core.SPECS if item.key == "D")
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

    rows = []
    generated = {}
    selected = {}
    for fraction in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        payload, combined, selections = run_fraction(frames, profiles, years, fraction)
        rows.append(payload)
        generated[fraction] = combined
        selected[fraction] = selections
    table = pd.DataFrame(rows)
    table["dev_ratio"] = table["dev_final_value"] / baseline_stats["dev"]["final_value"]
    table["holdout_ratio"] = table["holdout_final_value"] / baseline_stats["holdout"]["final_value"]
    table["full_ratio"] = table["full_final_value"] / baseline_stats["full"]["final_value"]
    table["passes"] = (
        table["dev_ratio"].ge(1.0)
        & table["holdout_ratio"].gt(1.0)
        & table["dev_win_rate"].ge(baseline_stats["dev"]["win_rate"] - 1e-12)
        & table["holdout_win_rate"].ge(baseline_stats["holdout"]["win_rate"] - 1e-12)
    )
    passed = table[table["passes"]]
    winner = passed.iloc[(len(passed) - 1) // 2] if len(passed) else table.iloc[0]
    fraction = float(winner["plateau_fraction"])
    table.to_csv(OUT / "sweep.csv", index=False, encoding="utf-8-sig")
    generated[fraction].to_csv(OUT / "winner_trades.csv", index=False, encoding="utf-8-sig")
    selected[fraction].to_csv(OUT / "winner_selections.csv", index=False, encoding="utf-8-sig")
    summary = {
        "method": "annual expanding-window near-optimal target plateau median",
        "minimum_training_trades": MIN_TRAIN_TRADES,
        "minimum_training_triggers": MIN_TRAIN_TRIGGERS,
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
