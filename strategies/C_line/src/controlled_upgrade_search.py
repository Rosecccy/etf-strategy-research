from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import take_profit_10_test as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "controlled_upgrade_search"
DEV_END = 2023
HOLDOUT_START = 2024


@dataclass(frozen=True)
class Rule:
    family: str
    a: float
    b: float = 0.0
    hold: int = 0

    @property
    def key(self) -> str:
        return f"{self.family}:a{self.a:+.3f}:b{self.b:+.3f}:h{self.hold}"


def candidate_rules() -> list[Rule]:
    rules: list[Rule] = []
    rules += [
        Rule("trailing", arm, trail, hold)
        for arm in (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30)
        for trail in (0.03, 0.05, 0.07, 0.10, 0.12, 0.15)
        for hold in (5, 10, 20, 30)
        if trail < arm + 0.05
    ]
    rules += [
        Rule("profit_floor", arm, floor, hold)
        for arm in (0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30)
        for floor in (0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15)
        for hold in (5, 10, 20, 30)
        if floor < arm
    ]
    rules += [
        Rule("loss_cut", threshold, 0.0, hold)
        for threshold in (-0.03, -0.05, -0.07, -0.10, -0.12, -0.15, -0.20)
        for hold in (5, 10, 20, 30)
    ]
    rules += [
        Rule(family, threshold, 0.0, hold)
        for family in ("ret20_weak", "ma20_weak")
        for threshold in (-0.15, -0.10, -0.07, -0.05, -0.03, 0.00)
        for hold in (10, 20, 30)
    ]
    rules += [
        Rule("fixed_target", target, 0.0, 0)
        for target in (0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40)
    ]
    return rules


def execute_early_exit(
    item: dict,
    data: pd.DataFrame,
    signal_pos: int,
    exit_pos: int,
    rule: Rule,
) -> bool:
    execute_pos = signal_pos + 1
    if execute_pos >= exit_pos or execute_pos >= len(data):
        return False
    row = data.iloc[execute_pos]
    item["exit_date_test"] = pd.Timestamp(row["date"])
    item["exit_close_test"] = float(row["close"])
    item["exit_reason_test"] = rule.key
    item["open_mark_test"] = False
    item["control_triggered"] = True
    item["control_signal_date"] = pd.Timestamp(data.iloc[signal_pos]["date"])
    return True


def apply_rule(
    trades: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    rule: Rule,
) -> pd.DataFrame:
    rows = []
    for _, trade in trades.iterrows():
        item = trade.to_dict()
        symbol = str(trade["symbol"])
        data = prices[symbol]
        dates = pd.DatetimeIndex(data["date"])
        closes = data["close"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        entry_pos = int(dates.searchsorted(pd.Timestamp(trade["entry_date"]), side="left"))
        end_date = core.AS_OF if bool(trade["original_open"]) else pd.Timestamp(trade["original_exit_date"])
        exit_pos = int(dates.searchsorted(end_date, side="left"))
        exit_pos = min(exit_pos, len(data) - 1)
        entry_price = float(trade["entry_close"])
        item["control_rule"] = rule.key
        item["control_triggered"] = False
        item["exit_date_test"] = pd.Timestamp(dates[exit_pos])
        item["exit_close_test"] = float(closes[exit_pos])
        item["open_mark_test"] = bool(trade["original_open"])
        item["exit_reason_test"] = "open_mark" if item["open_mark_test"] else "original_exit"

        peak_return = -np.inf
        peak_close = -np.inf
        start = min(entry_pos + max(rule.hold, 1), exit_pos)
        triggered = False
        for signal_pos in range(entry_pos + 1, exit_pos):
            current = closes[signal_pos]
            current_return = current / entry_price - 1.0
            peak_close = max(peak_close, current)
            peak_return = max(peak_return, current_return)
            if signal_pos < start:
                continue
            if rule.family == "trailing":
                hit = peak_return >= rule.a and peak_close > 0 and current / peak_close - 1.0 <= -rule.b
            elif rule.family == "profit_floor":
                hit = peak_return >= rule.a and current_return <= rule.b
            elif rule.family == "loss_cut":
                hit = current_return <= rule.a
            elif rule.family == "ret20_weak":
                hit = signal_pos >= 20 and current / closes[signal_pos - 20] - 1.0 <= rule.a
            elif rule.family == "ma20_weak":
                hit = signal_pos >= 19 and current / float(np.mean(closes[signal_pos - 19:signal_pos + 1])) - 1.0 <= rule.a
            elif rule.family == "fixed_target":
                target = entry_price * (1.0 + rule.a)
                if highs[signal_pos] >= target:
                    item["exit_date_test"] = pd.Timestamp(dates[signal_pos])
                    item["exit_close_test"] = max(target, float(opens[signal_pos]))
                    item["exit_reason_test"] = rule.key
                    item["open_mark_test"] = False
                    item["control_triggered"] = True
                    item["control_signal_date"] = pd.Timestamp(dates[signal_pos])
                    triggered = True
                    break
                hit = False
            else:
                raise ValueError(rule.family)
            if hit and execute_early_exit(item, data, signal_pos, exit_pos, rule):
                triggered = True
                break

        item["gross_return_test"] = float(item["exit_close_test"]) / entry_price - 1.0
        item["control_triggered"] = bool(triggered)
        item["take_profit_triggered"] = bool(triggered)
        rows.append(item)
    return pd.DataFrame(rows)


def subset_stats(frame: pd.DataFrame, kind: str) -> dict:
    years = pd.to_datetime(frame["entry_date"]).dt.year
    if kind == "dev":
        local = frame[years.le(DEV_END)].copy()
    elif kind == "holdout":
        local = frame[years.ge(HOLDOUT_START)].copy()
    else:
        local = frame.copy()
    stats, _, _ = core.simulate_account(local)
    return stats


def evaluate_line(spec: core.LineSpec, rules: list[Rule]) -> dict:
    trades = core.load_trades(spec)
    prices = core.load_prices(spec.project, set(trades["symbol"]))
    baseline = core.apply_take_profit(trades, prices, "baseline")
    base = {name: subset_stats(baseline, name) for name in ("full", "dev", "holdout")}
    rows = []
    generated: dict[str, pd.DataFrame] = {}
    for rule in rules:
        adjusted = apply_rule(trades, prices, rule)
        generated[rule.key] = adjusted
        stats = {name: subset_stats(adjusted, name) for name in ("full", "dev", "holdout")}
        row = {
            "rule": rule.key,
            "family": rule.family,
            "a": rule.a,
            "b": rule.b,
            "hold": rule.hold,
            "triggers": int(adjusted["control_triggered"].sum()),
        }
        for period, values in stats.items():
            for key in ("final_value", "win_rate", "max_drawdown", "avg_holding_days"):
                row[f"{period}_{key}"] = values[key]
            row[f"{period}_final_ratio"] = values["final_value"] / base[period]["final_value"]
            row[f"{period}_win_delta"] = values["win_rate"] - base[period]["win_rate"]
        row["strict_improvement"] = bool(
            row["dev_final_ratio"] > 1.0
            and row["dev_win_delta"] >= -1e-12
            and row["holdout_final_ratio"] > 1.0
            and row["holdout_win_delta"] >= -1e-12
            and row["full_final_ratio"] > 1.0
            and row["full_win_delta"] >= -1e-12
            and row["full_max_drawdown"] >= base["full"]["max_drawdown"] - 0.005
        )
        row["selection_score"] = (
            np.log(max(row["dev_final_ratio"], 1e-9))
            + 0.8 * row["dev_win_delta"]
            + 0.25 * np.log(max(row["full_final_ratio"], 1e-9))
            + 0.2 * row["full_win_delta"]
        )
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("selection_score", ascending=False).reset_index(drop=True)
    strict = table[table["strict_improvement"]].copy()
    if len(strict):
        winner_row = strict.sort_values(
            ["selection_score", "holdout_final_ratio"], ascending=False
        ).iloc[0]
        status = "strict_candidate"
    else:
        winner_row = table.iloc[0]
        status = "no_strict_candidate"
    winner = generated[str(winner_row["rule"])]
    winner_stats = {name: subset_stats(winner, name) for name in ("full", "dev", "holdout")}
    prefix = spec.key.lower()
    table.to_csv(OUT / f"{prefix}_candidates.csv", index=False, encoding="utf-8-sig")
    winner.to_csv(OUT / f"{prefix}_winner_trades.csv", index=False, encoding="utf-8-sig")
    return {
        "line": spec.key,
        "status": status,
        "strict_candidates": int(len(strict)),
        "winner_rule": str(winner_row["rule"]),
        "baseline": base,
        "winner": winner_stats,
    }


def main() -> None:
    core.configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    rules = candidate_rules()
    results = [evaluate_line(spec, rules) for spec in core.SPECS]
    summary = {
        "candidate_count": len(rules),
        "selection": "single-module controlled test; development through 2023; separate 2024-2026 check",
        "execution": "daily close signal, next-close execution; fixed target uses executable intraday limit",
        "results": results,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
