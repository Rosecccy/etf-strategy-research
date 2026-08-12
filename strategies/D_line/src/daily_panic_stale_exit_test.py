from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import daily_panic_rolling_test as base
from control_panic_age_test import portfolio_metrics


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "daily_panic_stale_exit_test"
DEV_META_YEARS = (2019, 2020, 2021, 2022, 2023)
HOLDOUT_YEARS = (2024, 2025, 2026)
WINDOWS: tuple[int | None, ...] = (5, None)
MIN_HOLDS = (20, 30, 40, 50, 60)
PEAK_CAPS = (0.01, 0.02, 0.03, 0.05)
RECOVERY_TARGETS = (0.0, 0.002, 0.005, 0.01)


@dataclass(frozen=True)
class Rule:
    min_hold: int
    peak_cap: float
    recovery_target: float

    @property
    def key(self) -> str:
        return f"h{self.min_hold}:cap{int(self.peak_cap * 1000)}:target{int(self.recovery_target * 1000)}"


RULES = tuple(
    Rule(min_hold, peak_cap, target)
    for min_hold in MIN_HOLDS
    for peak_cap in PEAK_CAPS
    for target in RECOVERY_TARGETS
)


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def build_formal(adjusted: pd.DataFrame, raw: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    rolling, _ = base.rolling_candidates(adjusted, raw, pool, None)
    early = adjusted[
        adjusted["test_year"].between(2016, 2018)
        & adjusted["fear_weighted"].ge(55)
        & adjusted["symbol_fear"].ge(70)
    ].copy()
    return pd.concat([early, rolling], ignore_index=True, sort=False)


def adjust_exits(frame: pd.DataFrame, raw: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    calendars = {
        symbol: group.sort_values("date").reset_index(drop=True)
        for symbol, group in raw.groupby("symbol", sort=False)
    }
    rows = []
    for _, trade in frame.iterrows():
        item = trade.to_dict()
        data = calendars.get(str(trade["symbol"]))
        if data is None or data.empty:
            rows.append(item)
            continue
        dates = pd.DatetimeIndex(data["date"])
        entry_date = pd.Timestamp(trade["entry_date"])
        entry_pos = int(dates.searchsorted(entry_date, side="left"))
        if entry_pos >= len(data):
            rows.append(item)
            continue
        original_exit = pd.to_datetime(trade.get("exit_date"), errors="coerce")
        end_pos = int(dates.searchsorted(original_exit, side="left")) if pd.notna(original_exit) else len(data) - 1
        end_pos = min(end_pos, len(data) - 1)
        entry_price = float(trade["entry_close"])
        closes = pd.to_numeric(data["close"], errors="coerce").to_numpy(dtype=float)
        peak = -np.inf
        trigger_pos = None
        for pos in range(entry_pos, end_pos + 1):
            current_return = closes[pos] / entry_price - 1.0
            peak = max(peak, current_return)
            held = pos - entry_pos
            if held < rule.min_hold:
                continue
            if peak < rule.peak_cap and current_return >= rule.recovery_target:
                candidate = pos + 1
                if candidate <= end_pos:
                    trigger_pos = candidate
                break
        if trigger_pos is not None:
            exit_date = pd.Timestamp(dates[trigger_pos])
            exit_close = float(closes[trigger_pos])
            item["exit_date"] = exit_date
            item["exit_close"] = exit_close
            item["status"] = "closed"
            item["return_rate"] = exit_close / entry_price - 1.0
            item["stale_exit_triggered"] = True
            item["stale_exit_rule"] = rule.key
            item["stale_signal_date"] = pd.Timestamp(dates[trigger_pos - 1])
            item["stale_signal_peak_return"] = peak
            item["stale_signal_current_return"] = closes[trigger_pos - 1] / entry_price - 1.0
        else:
            item["stale_exit_triggered"] = False
            item["stale_exit_rule"] = rule.key
            item["stale_signal_date"] = pd.NaT
            item["stale_signal_peak_return"] = np.nan
            item["stale_signal_current_return"] = np.nan
        rows.append(item)
    return pd.DataFrame(rows)


def score(stat: dict) -> float:
    return (
        math.log(max(float(stat["final_value"]), 1.0) / 10_000.0)
        + 0.75 * float(np.nan_to_num(stat["win_rate"], nan=0.0))
        + 0.80 * float(stat["max_drawdown"])
    )


def neighbor_count(rule: Rule, eligible: set[str]) -> int:
    hi = MIN_HOLDS.index(rule.min_hold)
    ci = PEAK_CAPS.index(rule.peak_cap)
    ti = RECOVERY_TARGETS.index(rule.recovery_target)
    count = 0
    for other in RULES:
        if other.key not in eligible:
            continue
        distance = (
            abs(MIN_HOLDS.index(other.min_hold) - hi)
            + abs(PEAK_CAPS.index(other.peak_cap) - ci)
            + abs(RECOVERY_TARGETS.index(other.recovery_target) - ti)
        )
        if distance <= 1:
            count += 1
    return count


def choose_rule(
    formal: pd.DataFrame,
    logs: dict[str, pd.DataFrame],
    raw: pd.DataFrame,
    pool: pd.DataFrame,
    year: int,
    window: int | None,
) -> tuple[Rule | None, dict]:
    start = 2016 if window is None else max(2016, year - window)
    years = set(range(start, year))
    cutoff = pd.Timestamp(year=year, month=1, day=1)

    def history(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[
            frame["test_year"].isin(years)
            & frame["status"].eq("closed")
            & pd.to_datetime(frame["exit_date"]).lt(cutoff)
        ].copy()

    _, baseline = portfolio_metrics(history(formal), raw, pool)
    if int(baseline["closed_trades"]) < 6:
        return None, {"fallback": True, **baseline}
    rows = []
    for rule in RULES:
        _, stat = portfolio_metrics(history(logs[rule.key]), raw, pool)
        eligible = bool(
            int(stat["closed_trades"]) >= max(5, math.ceil(int(baseline["closed_trades"]) * 0.70))
            and float(stat["final_value"]) > float(baseline["final_value"]) * 1.01
            and float(stat["win_rate"]) > float(baseline["win_rate"]) + 0.005
            and float(stat["max_drawdown"]) >= float(baseline["max_drawdown"]) - 0.02
        )
        rows.append({"rule": rule, "key": rule.key, "eligible": eligible, "score": score(stat), **stat})
    table = pd.DataFrame(rows)
    eligible = set(table.loc[table["eligible"], "key"])
    if not eligible:
        return None, {"fallback": True, **baseline}
    table["neighbors"] = table["rule"].map(lambda item: neighbor_count(item, eligible))
    stable = table[table["eligible"] & table["neighbors"].ge(2)].sort_values(
        ["score", "final_value", "win_rate"], ascending=False
    )
    if stable.empty:
        return None, {"fallback": True, **baseline}
    winner = stable.iloc[0]
    return winner["rule"], winner.drop(labels=["rule"]).to_dict()


def rolling(
    formal: pd.DataFrame,
    logs: dict[str, pd.DataFrame],
    raw: pd.DataFrame,
    pool: pd.DataFrame,
    window: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    choices = []
    for year in range(2019, 2027):
        rule, audit = choose_rule(formal, logs, raw, pool, year, window)
        source = formal if rule is None else logs[rule.key]
        current = source[source["test_year"].eq(year)].copy()
        current["selected_stale_exit"] = "BASE_EXIT" if rule is None else rule.key
        parts.append(current)
        choices.append(
            {
                "year": year,
                "window": "all" if window is None else window,
                "train_start": 2016 if window is None else max(2016, year - window),
                "train_end": year - 1,
                "rule": "BASE_EXIT" if rule is None else rule.key,
                "train_score": audit.get("score"),
                "train_trades": audit.get("closed_trades"),
                "neighbors": audit.get("neighbors", 0),
            }
        )
    return pd.concat(parts, ignore_index=True, sort=False), pd.DataFrame(choices)


def evaluate(frame: pd.DataFrame, raw: pd.DataFrame, pool: pd.DataFrame, years: tuple[int, ...]) -> tuple[pd.DataFrame, dict]:
    return portfolio_metrics(frame[frame["test_year"].isin(years)].copy(), raw, pool)


def main() -> None:
    configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    adjusted, raw, pool = base.prepare()
    formal = build_formal(adjusted, raw, pool)
    logs = {rule.key: adjust_exits(formal, raw, rule) for rule in RULES}
    _, base_dev = evaluate(formal, raw, pool, DEV_META_YEARS)
    base_holdout_log, base_holdout = evaluate(formal, raw, pool, HOLDOUT_YEARS)
    rows = []
    generated = {}
    for window in WINDOWS:
        key = "all" if window is None else str(window)
        candidates, choices = rolling(formal, logs, raw, pool, window)
        _, dev = evaluate(candidates, raw, pool, DEV_META_YEARS)
        holdout_log, holdout = evaluate(candidates, raw, pool, HOLDOUT_YEARS)
        generated[key] = (choices, holdout_log)
        rows.append({"window": key, **{f"dev_{k}": v for k, v in dev.items()}, **{f"holdout_{k}": v for k, v in holdout.items()}})
    table = pd.DataFrame(rows)
    dev_ok = table[
        table["dev_final_value"].gt(float(base_dev["final_value"]))
        & table["dev_win_rate"].gt(float(base_dev["win_rate"]))
    ].copy()
    if dev_ok.empty:
        chosen = "all"
    else:
        dev_ok["score"] = np.log(dev_ok["dev_final_value"] / 10_000.0) + 0.75 * dev_ok["dev_win_rate"] + 0.80 * dev_ok["dev_max_drawdown"]
        chosen = str(dev_ok.sort_values("score", ascending=False).iloc[0]["window"])
    choices, winner_log = generated[chosen]
    winner_row = table[table["window"].eq(chosen)].iloc[0]
    winner = {key.removeprefix("holdout_"): value for key, value in winner_row.to_dict().items() if key.startswith("holdout_")}
    robust = int(
        (
            table["holdout_final_value"].gt(float(base_holdout["final_value"]))
            & table["holdout_win_rate"].gt(float(base_holdout["win_rate"]))
            & table["holdout_max_drawdown"].ge(float(base_holdout["max_drawdown"]))
        ).sum()
    )
    promoted = bool(
        float(winner["final_value"]) > float(base_holdout["final_value"])
        and float(winner["win_rate"]) > float(base_holdout["win_rate"])
        and float(winner["max_drawdown"]) >= float(base_holdout["max_drawdown"])
        and int(winner["closed_trades"]) >= 6
        and robust >= 2
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test": "D2 annual past-only stale-position recovery exit",
        "execution": "exit condition after close, execute at next trading-day close",
        "development_selected_window": chosen,
        "baseline_development": base_dev,
        "baseline_holdout": base_holdout,
        "winner_holdout": winner,
        "holdout_neighbor_windows_both_improved": robust,
        "promoted": promoted,
    }
    table.to_csv(OUT / "variants.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(OUT / "winner_choices.csv", index=False, encoding="utf-8-sig")
    winner_log.to_csv(OUT / "winner_holdout_trades.csv", index=False, encoding="utf-8-sig")
    base_holdout_log.to_csv(OUT / "baseline_holdout_trades.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
