from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import controlled_upgrade_search as search
import staged_upgrade_refine as staged
import take_profit_10_test as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "walkforward_overlay"
MIN_TRAIN_TRADES = 12
MIN_TRAIN_TRIGGERS = 2


def effective_triggers(frame: pd.DataFrame) -> pd.Series:
    stage1 = frame.get("stage1_triggered", pd.Series(False, index=frame.index)).astype(bool)
    stage2 = frame.get("control_triggered", pd.Series(False, index=frame.index)).astype(bool)
    return stage1 | stage2


def candidate_frames(
    spec: core.LineSpec,
    trades: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    baseline = core.apply_take_profit(trades, prices, "baseline")
    frames = {"baseline": baseline}
    if spec.key == "C":
        stage1_rule = search.Rule("ret20_weak", -0.10, 0.0, 20)
        first = search.apply_rule(trades, prices, stage1_rule)
        first["stage1_triggered"] = first["control_triggered"]
        first["stage1_rule"] = stage1_rule.key
        frames[stage1_rule.key] = first
        source = staged.rebase(first)
        for arm in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
            for trail in (0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10):
                for hold in (20, 25, 30, 35, 40):
                    rule = search.Rule("trailing", arm, trail, hold)
                    adjusted = search.apply_rule(source, prices, rule)
                    adjusted["stage1_triggered"] = source["stage1_triggered"].to_numpy()
                    adjusted["stage1_rule"] = stage1_rule.key
                    frames[f"{stage1_rule.key}+{rule.key}"] = adjusted
    elif spec.key == "S":
        for arm in (0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115, 0.120):
            for floor in (0.060, 0.070, 0.075, 0.080, 0.085, 0.090, 0.095, 0.100):
                if floor >= arm:
                    continue
                for hold in (25, 28, 30, 32, 35, 40):
                    rule = search.Rule("profit_floor", arm, floor, hold)
                    frames[rule.key] = search.apply_rule(trades, prices, rule)
    else:
        for target in np.arange(0.14, 0.251, 0.01):
            rule = search.Rule("fixed_target", float(target), 0.0, 0)
            frames[rule.key] = search.apply_rule(trades, prices, rule)
    return frames


def training_profile(frame: pd.DataFrame, years: list[int]) -> dict[int, dict]:
    _, detail, _ = core.simulate_account(frame)
    detail_years = pd.to_datetime(detail["entry_date"]).dt.year
    frame_years = pd.to_datetime(frame["entry_date"]).dt.year
    result = {}
    for year in years:
        local = detail[detail_years.lt(year)].copy()
        source = frame[frame_years.lt(year)].copy()
        if local.empty:
            result[year] = {
                "trades": 0,
                "final_value": core.INITIAL_CAPITAL,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
                "trigger_count": 0,
            }
            continue
        returns = local["account_return_test"].astype(float)
        equity = core.INITIAL_CAPITAL * (1.0 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        closed = local[~local["open_mark_test"].astype(bool)]
        result[year] = {
            "trades": int(len(local)),
            "final_value": float(local.iloc[-1]["cash_after_test"]),
            "win_rate": float((closed["account_return_test"] > 0).mean()) if len(closed) else 0.0,
            "max_drawdown": float(drawdown.min()),
            "trigger_count": int(effective_triggers(source).sum()),
        }
    return result


def choose_rule(
    profiles: dict[str, dict[int, dict]],
    year: int,
) -> tuple[str, dict, list[dict]]:
    baseline = profiles["baseline"][year]
    rows = []
    for name, profile in profiles.items():
        current = profile[year]
        final_ratio = current["final_value"] / baseline["final_value"]
        win_delta = current["win_rate"] - baseline["win_rate"]
        dd_delta = current["max_drawdown"] - baseline["max_drawdown"]
        eligible = bool(
            name != "baseline"
            and current["trades"] >= MIN_TRAIN_TRADES
            and current["trigger_count"] >= MIN_TRAIN_TRIGGERS
            and final_ratio > 1.0
            and win_delta >= -1e-12
            and dd_delta >= -0.005
        )
        score = (
            np.log(max(final_ratio, 1e-12))
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
                "train_final_ratio": final_ratio,
                "train_win_delta": win_delta,
                "train_drawdown_delta": dd_delta,
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    winner = max(eligible, key=lambda item: item["score"]) if eligible else next(
        row for row in rows if row["rule"] == "baseline"
    )
    return str(winner["rule"]), winner, rows


def validate_line(
    spec: core.LineSpec,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades = core.load_trades(spec)
    prices = core.load_prices(spec.project, set(trades["symbol"]))
    frames = candidate_frames(spec, trades, prices)
    years = sorted(pd.to_datetime(trades["entry_date"]).dt.year.unique())
    profiles = {name: training_profile(frame, years) for name, frame in frames.items()}
    selected_parts = []
    selection_rows = []
    grid_rows = []
    for year in years:
        rule, selection, rows = choose_rule(profiles, int(year))
        selected = frames[rule][pd.to_datetime(frames[rule]["entry_date"]).dt.year.eq(year)].copy()
        selected["walkforward_rule"] = rule
        selected["walkforward_test_year"] = year
        selected_parts.append(selected)
        selection_rows.append(selection)
        grid_rows.extend(rows)
    combined = pd.concat(selected_parts, ignore_index=True).sort_values(["entry_date", "symbol"])
    baseline = frames["baseline"].copy()
    base_stats, _, _ = core.simulate_account(baseline)
    wf_stats, _, _ = core.simulate_account(combined)
    base_holdout = search.subset_stats(baseline, "holdout")
    wf_holdout = search.subset_stats(combined, "holdout")
    payload = {
        "line": spec.key,
        "candidate_rules": len(frames),
        "years": [int(years[0]), int(years[-1])],
        "baseline": {"full": base_stats, "holdout": base_holdout},
        "walkforward": {"full": wf_stats, "holdout": wf_holdout},
        "full_final_ratio": wf_stats["final_value"] / base_stats["final_value"],
        "holdout_final_ratio": wf_holdout["final_value"] / base_holdout["final_value"],
        "selected_nonbaseline_years": int(
            sum(row["rule"] != "baseline" for row in selection_rows)
        ),
    }
    return payload, combined, pd.DataFrame(selection_rows), pd.DataFrame(grid_rows)


def main() -> None:
    core.configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in core.SPECS:
        payload, trades, selections, grid = validate_line(spec)
        results.append(payload)
        prefix = spec.key.lower()
        trades.to_csv(OUT / f"{prefix}_trades.csv", index=False, encoding="utf-8-sig")
        selections.to_csv(OUT / f"{prefix}_selections.csv", index=False, encoding="utf-8-sig")
        grid.to_csv(OUT / f"{prefix}_grid.csv", index=False, encoding="utf-8-sig")
    summary = {
        "method": "annual expanding-window overlay selection using only prior-year trades",
        "minimum_training_trades": MIN_TRAIN_TRADES,
        "minimum_training_triggers": MIN_TRAIN_TRIGGERS,
        "results": results,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
