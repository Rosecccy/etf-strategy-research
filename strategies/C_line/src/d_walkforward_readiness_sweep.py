from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import controlled_upgrade_search as search
import take_profit_10_test as core
import walkforward_overlay_validation as wf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "d_walkforward_readiness"
TARGET_CAP = 0.20


def choose(
    profiles: dict[str, dict[int, dict]],
    year: int,
    min_trades: int,
    min_triggers: int,
) -> tuple[str, dict]:
    baseline = profiles["baseline"][year]
    rows = []
    for name, profile in profiles.items():
        current = profile[year]
        ratio = current["final_value"] / baseline["final_value"]
        win_delta = current["win_rate"] - baseline["win_rate"]
        dd_delta = current["max_drawdown"] - baseline["max_drawdown"]
        eligible = bool(
            name != "baseline"
            and current["trades"] >= min_trades
            and current["trigger_count"] >= min_triggers
            and ratio > 1.0
            and win_delta >= -1e-12
            and dd_delta >= -0.005
        )
        score = (
            np.log(max(ratio, 1e-12))
            + 0.60 * win_delta
            + 0.25 * min(dd_delta, 0.20)
            - 0.002 * max(current["trigger_count"] - 12, 0)
        )
        rows.append(
            {
                "year": year,
                "rule": name,
                "eligible": eligible,
                "score": score,
                "train_trades": current["trades"],
                "train_triggers": current["trigger_count"],
                "train_ratio": ratio,
            }
        )
    candidates = [row for row in rows if row["eligible"]]
    winner = max(candidates, key=lambda row: row["score"]) if candidates else next(
        row for row in rows if row["rule"] == "baseline"
    )
    return str(winner["rule"]), winner


def run_policy(
    frames: dict[str, pd.DataFrame],
    profiles: dict[str, dict[int, dict]],
    years: list[int],
    min_trades: int,
    min_triggers: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    parts = []
    selections = []
    for year in years:
        rule, selection = choose(profiles, year, min_trades, min_triggers)
        part = frames[rule][pd.to_datetime(frames[rule]["entry_date"]).dt.year.eq(year)].copy()
        part["walkforward_rule"] = rule
        part["walkforward_test_year"] = year
        parts.append(part)
        selections.append(selection)
    combined = pd.concat(parts, ignore_index=True).sort_values(["entry_date", "symbol"])
    stats = {
        "full": search.subset_stats(combined, "full"),
        "dev": search.subset_stats(combined, "dev"),
        "holdout": search.subset_stats(combined, "holdout"),
    }
    payload = {
        "min_trades": min_trades,
        "min_triggers": min_triggers,
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
    all_frames = wf.candidate_frames(spec, trades, prices)
    frames = {"baseline": all_frames["baseline"]}
    for name, frame in all_frames.items():
        if name == "baseline":
            continue
        target = float(name.split(":a", 1)[1].split(":", 1)[0])
        if target <= TARGET_CAP + 1e-12:
            frames[name] = frame
    years = sorted(pd.to_datetime(trades["entry_date"]).dt.year.unique())
    profiles = {name: wf.training_profile(frame, years) for name, frame in frames.items()}
    baseline = frames["baseline"]
    baseline_stats = {
        period: search.subset_stats(baseline, period)
        for period in ("full", "dev", "holdout")
    }

    rows = []
    generated = {}
    selections = {}
    stage1_values = [6, 8, 10, 12, 15]
    for min_trades in stage1_values:
        payload, combined, selected = run_policy(frames, profiles, years, min_trades, 2)
        payload["stage"] = "min_trades"
        rows.append(payload)
        generated[(min_trades, 2)] = combined
        selections[(min_trades, 2)] = selected

    stage1 = pd.DataFrame(rows)
    stage1["dev_ratio"] = stage1["dev_final_value"] / baseline_stats["dev"]["final_value"]
    stage1["holdout_ratio"] = stage1["holdout_final_value"] / baseline_stats["holdout"]["final_value"]
    viable = stage1[
        stage1["dev_ratio"].gt(1.0)
        & stage1["dev_win_rate"].ge(baseline_stats["dev"]["win_rate"] - 1e-12)
    ]
    selected_min_trades = int(
        viable.sort_values("dev_ratio", ascending=False).iloc[0]["min_trades"]
        if len(viable)
        else 12
    )

    for min_triggers in (1, 2, 3, 4):
        key = (selected_min_trades, min_triggers)
        if key in generated:
            continue
        payload, combined, selected = run_policy(
            frames, profiles, years, selected_min_trades, min_triggers
        )
        payload["stage"] = "min_triggers"
        rows.append(payload)
        generated[key] = combined
        selections[key] = selected

    table = pd.DataFrame(rows)
    table["dev_ratio"] = table["dev_final_value"] / baseline_stats["dev"]["final_value"]
    table["holdout_ratio"] = table["holdout_final_value"] / baseline_stats["holdout"]["final_value"]
    table["dev_win_delta"] = table["dev_win_rate"] - baseline_stats["dev"]["win_rate"]
    table["holdout_win_delta"] = table["holdout_win_rate"] - baseline_stats["holdout"]["win_rate"]
    table["passes_both"] = (
        table["dev_ratio"].gt(1.0)
        & table["holdout_ratio"].gt(1.0)
        & table["dev_win_delta"].ge(-1e-12)
        & table["holdout_win_delta"].ge(-1e-12)
    )
    passed = table[table["passes_both"]]
    if len(passed):
        winner = passed.sort_values(["dev_ratio", "holdout_ratio"], ascending=False).iloc[0]
        key = (int(winner["min_trades"]), int(winner["min_triggers"]))
        generated[key].to_csv(OUT / "winner_trades.csv", index=False, encoding="utf-8-sig")
        selections[key].to_csv(OUT / "winner_selections.csv", index=False, encoding="utf-8-sig")
        winner_payload = winner.to_dict()
    else:
        winner_payload = None
    table.to_csv(OUT / "sweep.csv", index=False, encoding="utf-8-sig")
    summary = {
        "target_cap": TARGET_CAP,
        "baseline": baseline_stats,
        "stage1_selected_min_trades_from_dev": selected_min_trades,
        "winner": winner_payload,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
