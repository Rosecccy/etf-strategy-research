from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import controlled_upgrade_search as search
import take_profit_10_test as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "staged_upgrade_refine"


def rebase(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["original_exit_date"] = pd.to_datetime(result["exit_date_test"], errors="coerce")
    result["original_exit_close"] = pd.to_numeric(result["exit_close_test"], errors="coerce")
    result["original_open"] = result["open_mark_test"].astype(bool)
    return result


def stats(frame: pd.DataFrame) -> dict[str, dict]:
    return {name: search.subset_stats(frame, name) for name in ("full", "dev", "holdout")}


def row_for(rule: search.Rule, adjusted: pd.DataFrame, baseline: dict, stage: str) -> dict:
    result = {
        "stage": stage,
        "rule": rule.key,
        "family": rule.family,
        "a": rule.a,
        "b": rule.b,
        "hold": rule.hold,
        "triggers": int(adjusted["control_triggered"].sum()),
    }
    current = stats(adjusted)
    for period in ("full", "dev", "holdout"):
        result[f"{period}_final_value"] = current[period]["final_value"]
        result[f"{period}_final_ratio"] = current[period]["final_value"] / baseline[period]["final_value"]
        result[f"{period}_win_rate"] = current[period]["win_rate"]
        result[f"{period}_win_delta"] = current[period]["win_rate"] - baseline[period]["win_rate"]
        result[f"{period}_max_drawdown"] = current[period]["max_drawdown"]
    result["strict"] = bool(
        result["full_final_ratio"] > 1.0
        and result["dev_final_ratio"] > 1.0
        and result["holdout_final_ratio"] > 1.0
        and result["full_win_delta"] >= -1e-12
        and result["dev_win_delta"] >= -1e-12
        and result["holdout_win_delta"] >= -1e-12
        and result["full_max_drawdown"] >= baseline["full"]["max_drawdown"] - 0.005
    )
    result["dev_score"] = (
        np.log(max(result["dev_final_ratio"], 1e-9))
        + 0.8 * result["dev_win_delta"]
        + 0.35 * np.log(max(result["full_final_ratio"], 1e-9))
    )
    return result


def evaluate_grid(
    spec: core.LineSpec,
    rules: list[search.Rule],
    stage_rule: search.Rule | None = None,
    preferred_rule: search.Rule | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    trades = core.load_trades(spec)
    prices = core.load_prices(spec.project, set(trades["symbol"]))
    baseline_frame = core.apply_take_profit(trades, prices, "baseline")
    baseline = stats(baseline_frame)
    source = trades
    stage_name = "single"
    stage_stats = None
    if stage_rule is not None:
        first = search.apply_rule(trades, prices, stage_rule)
        first["stage1_triggered"] = first["control_triggered"]
        first["stage1_rule"] = stage_rule.key
        source = rebase(first)
        stage_name = f"{stage_rule.key}+stage2"
        stage_stats = stats(first)

    rows = []
    generated = {}
    for rule in rules:
        adjusted = search.apply_rule(source, prices, rule)
        if stage_rule is not None:
            adjusted["stage1_triggered"] = source["stage1_triggered"].to_numpy()
            adjusted["stage1_rule"] = stage_rule.key
        rows.append(row_for(rule, adjusted, baseline, stage_name))
        generated[rule.key] = adjusted
    table = pd.DataFrame(rows).sort_values("dev_score", ascending=False).reset_index(drop=True)
    strict = table[table["strict"]].copy()
    preferred = (
        strict[strict["rule"].eq(preferred_rule.key)]
        if preferred_rule is not None
        else strict.iloc[0:0]
    )
    winner_row = preferred.iloc[0] if len(preferred) else (strict.iloc[0] if len(strict) else table.iloc[0])
    winner = generated[str(winner_row["rule"])]
    if spec.key == "C":
        nearby = strict[
            strict["a"].sub(float(winner_row["a"])).abs().le(0.051)
            & strict["b"].sub(float(winner_row["b"])).abs().le(0.011)
            & strict["hold"].sub(int(winner_row["hold"])).abs().le(5)
        ]
    elif spec.key == "S":
        nearby = strict[
            strict["a"].sub(float(winner_row["a"])).abs().le(0.006)
            & strict["b"].sub(float(winner_row["b"])).abs().le(0.006)
            & strict["hold"].sub(int(winner_row["hold"])).abs().le(2)
        ]
    else:
        nearby = strict[strict["a"].sub(float(winner_row["a"])).abs().le(0.011)]
    payload = {
        "line": spec.key,
        "stage1_rule": stage_rule.key if stage_rule else None,
        "strict_candidates": int(len(strict)),
        "strict_nearby_candidates": int(len(nearby)),
        "winner_rule": str(winner_row["rule"]),
        "baseline": baseline,
        "stage1": stage_stats,
        "winner": stats(winner),
    }
    return payload, table, winner


def main() -> None:
    core.configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    specs = {spec.key: spec for spec in core.SPECS}

    c_stage1 = search.Rule("ret20_weak", -0.10, 0.0, 20)
    c_rules = [
        search.Rule("trailing", arm, trail, hold)
        for arm in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
        for trail in (0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10)
        for hold in (20, 25, 30, 35, 40)
    ]
    s_rules = [
        search.Rule("profit_floor", arm, floor, hold)
        for arm in (0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115, 0.120)
        for floor in (0.060, 0.070, 0.075, 0.080, 0.085, 0.090, 0.095, 0.100)
        for hold in (25, 28, 30, 32, 35, 40)
        if floor < arm
    ]
    d_rules = [
        search.Rule("fixed_target", target, 0.0, 0)
        for target in np.arange(0.14, 0.251, 0.01)
    ]

    outputs = []
    for key, rules, stage1, preferred in (
        ("C", c_rules, c_stage1, search.Rule("trailing", 0.25, 0.05, 40)),
        ("S", s_rules, None, search.Rule("profit_floor", 0.10, 0.08, 30)),
        ("D", d_rules, None, search.Rule("fixed_target", 0.20, 0.0, 0)),
    ):
        payload, table, winner = evaluate_grid(specs[key], rules, stage1, preferred)
        outputs.append(payload)
        prefix = key.lower()
        table.to_csv(OUT / f"{prefix}_grid.csv", index=False, encoding="utf-8-sig")
        winner.to_csv(OUT / f"{prefix}_winner_trades.csv", index=False, encoding="utf-8-sig")

    summary = {
        "selection": "staged control-variable refinement; winner ranked on development score and required to pass full/dev/2024-2026 gates",
        "results": outputs,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
