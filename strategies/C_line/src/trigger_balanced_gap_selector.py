from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import controlled_upgrade_search as stats_mod
import cross_line_gap_overlay as gap
import take_profit_10_test as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "trigger_balanced_gap"


def choose(
    target: pd.DataFrame,
    candidates: dict[tuple[str, ...], pd.DataFrame],
    year: int,
) -> tuple[tuple[str, ...] | None, dict]:
    base_train = target[pd.to_datetime(target["entry_date"]).dt.year.lt(year)].copy()
    if len(base_train) < 12:
        return None, {"test_year": year, "priority": "baseline", "reason": "insufficient_history"}
    base = core.simulate_account(base_train)[0]
    passing = []
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
            + 0.50 * win_delta
            + 0.10 * np.log(max(trigger_ratio, 1e-12))
            + 0.20 * min(dd_delta, 0.10)
        )
        passes = bool(
            added >= 2
            and trigger_ratio >= 1.10
            and final_ratio >= 0.995
            and win_delta >= -0.02
            and dd_delta >= -0.01
            and score > 0.0
        )
        if passes:
            passing.append(
                {
                    "test_year": year,
                    "priority": ">".join(priority),
                    "priority_tuple": priority,
                    "reason": "past_only_balanced_score",
                    "train_base_trades": len(base_train),
                    "train_added_trades": added,
                    "train_trigger_ratio": trigger_ratio,
                    "train_final_ratio": final_ratio,
                    "train_win_delta": win_delta,
                    "train_dd_delta": dd_delta,
                    "score": score,
                }
            )
    if not passing:
        return None, {"test_year": year, "priority": "baseline", "reason": "no_balanced_candidate"}
    winner = max(passing, key=lambda row: row["score"])
    priority = winner.pop("priority_tuple")
    return priority, winner


def evaluate(
    line: str,
    frames: dict[str, pd.DataFrame],
    prices: dict[tuple[str, str], pd.DataFrame],
) -> dict:
    target = frames[line]
    candidates = {
        priority: gap.build_schedule(line, target, frames, priority, prices)
        for priority in gap.overlay_options(line)
    }
    years = sorted(pd.to_datetime(target["entry_date"]).dt.year.unique())
    parts = []
    selections = []
    for year in years:
        priority, record = choose(target, candidates, int(year))
        source = target if priority is None else candidates[priority]
        local = source[pd.to_datetime(source["entry_date"]).dt.year.eq(year)].copy()
        local["walkforward_trigger_priority"] = "baseline" if priority is None else ">".join(priority)
        parts.append(local)
        selections.append(record)
    winner = pd.concat(parts, ignore_index=True).sort_values(["entry_date", "is_gap_overlay", "symbol"])
    baseline = {period: stats_mod.subset_stats(target, period) for period in ("full", "holdout")}
    current = {period: stats_mod.subset_stats(winner, period) for period in ("full", "holdout")}
    start = int(pd.to_datetime(target["entry_date"]).dt.year.min())
    year_count = 2026 - start + 1
    payload = {
        "line": line,
        "baseline": baseline,
        "winner": current,
        "baseline_triggers": len(target),
        "winner_triggers": len(winner),
        "added_triggers": int(winner["is_gap_overlay"].fillna(False).sum()),
        "trigger_ratio": len(winner) / len(target),
        "baseline_avg_triggers_per_year": len(target) / year_count,
        "winner_avg_triggers_per_year": len(winner) / year_count,
        "full_ratio": current["full"]["final_value"] / baseline["full"]["final_value"],
        "holdout_ratio": current["holdout"]["final_value"] / baseline["holdout"]["final_value"],
    }
    winner.to_csv(OUT / f"{line.lower()}_trades.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selections).to_csv(OUT / f"{line.lower()}_selections.csv", index=False, encoding="utf-8-sig")
    return payload


def main() -> None:
    core.configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {line: gap.load_frame(line) for line in ("C", "S", "D")}
    prices = gap.load_all_prices(frames)
    payload = {
        "method": "annual past-only gap overlay selected by return, win rate, trigger rate and drawdown",
        "score": "log(final_ratio)+0.50*win_delta+0.10*log(trigger_ratio)+0.20*dd_delta",
        "constraints": {
            "trigger_ratio_min": 1.10,
            "final_ratio_min": 0.995,
            "win_delta_min": -0.02,
            "drawdown_delta_min": -0.01,
        },
        "results": [evaluate(line, frames, prices) for line in ("C", "S", "D")],
    }
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
