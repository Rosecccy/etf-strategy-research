from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stale_recovery_exit_overlay_test import fill_trade_prices, metrics, read_trades


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
D_SRC = PROJECT / "D" / "src"
if str(D_SRC) not in sys.path:
    sys.path.insert(0, str(D_SRC))

import fear_greed_oos as fg  # noqa: E402


OUT = ROOT / "fit" / "deep_downtrend_recovery_exit_test"
C_TRADES = ROOT / "fit" / "hybrid_formal" / "final_trades.csv"
S_TRADES = PROJECT / "S" / "fit" / "selector" / "final_trades.csv"
DEV_META_YEARS = {2019, 2020, 2021, 2022, 2023}
HOLDOUT_YEARS = {2024, 2025, 2026}
WINDOWS: tuple[int | None, ...] = (3, 5, None)
TREND_LIMITS = (-0.05, -0.08, -0.10, -0.12, -0.15, -0.20)
MIN_HOLDS = (5, 7, 10, 15, 20)
TROUGH_LIMITS = (-0.02, -0.03, -0.05, -0.08)
RECOVERY_TARGETS = (-0.03, -0.02, -0.01, 0.0, 0.005)


@dataclass(frozen=True)
class Rule:
    scope: str
    trend_limit: float
    min_hold: int
    trough_limit: float
    recovery_target: float

    @property
    def key(self) -> str:
        return (
            f"{self.scope}:trend{int(self.trend_limit * 100)}:h{self.min_hold}:"
            f"trough{int(self.trough_limit * 100)}:recover{int(self.recovery_target * 1000)}"
        )


RULES = tuple(
    Rule(scope, trend, hold, trough, recovery)
    for scope in ("all", "main")
    for trend in TREND_LIMITS
    for hold in MIN_HOLDS
    for trough in TROUGH_LIMITS
    for recovery in RECOVERY_TARGETS
)


def targeted(source: object, scope: str) -> bool:
    return scope == "all" or str(source) == "主策略"


def attach_pre_entry_ret60(frame: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    calendars = {
        symbol: group.sort_values("date").reset_index(drop=True)
        for symbol, group in raw.groupby("symbol", sort=False)
    }
    values = []
    for _, trade in result.iterrows():
        data = calendars.get(str(trade["symbol"]))
        if data is None:
            values.append(np.nan)
            continue
        prior = data[data["date"].lt(pd.Timestamp(trade["entry_date"]))]
        if len(prior) < 61:
            values.append(np.nan)
            continue
        values.append(float(prior.iloc[-1]["close"] / prior.iloc[-61]["close"] - 1.0))
    result["pre_entry_ret60"] = values
    return result


def adjust(frame: pd.DataFrame, raw: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    calendars = {
        symbol: group.sort_values("date").reset_index(drop=True)
        for symbol, group in raw.groupby("symbol", sort=False)
    }
    rows = []
    for _, trade in frame.iterrows():
        item = trade.to_dict()
        item["deep_recovery_rule"] = rule.key
        item["deep_recovery_triggered"] = False
        trend = float(trade["pre_entry_ret60"]) if pd.notna(trade["pre_entry_ret60"]) else np.nan
        if not targeted(trade.get("source"), rule.scope) or not np.isfinite(trend) or trend > rule.trend_limit:
            rows.append(item)
            continue
        data = calendars.get(str(trade["symbol"]))
        if data is None:
            rows.append(item)
            continue
        dates = pd.DatetimeIndex(data["date"])
        entry_pos = int(dates.searchsorted(pd.Timestamp(trade["entry_date"]), side="left"))
        exit_pos = int(dates.searchsorted(pd.Timestamp(trade["exit_date"]), side="left"))
        if entry_pos >= len(data) or exit_pos >= len(data) or exit_pos <= entry_pos:
            rows.append(item)
            continue
        closes = pd.to_numeric(data["close"], errors="coerce").to_numpy(dtype=float)
        entry_price = float(trade["entry_close"])
        trough = 0.0
        execute_pos = None
        signal_pos = None
        for pos in range(entry_pos, exit_pos):
            current = closes[pos] / entry_price - 1.0
            trough = min(trough, current)
            held = pos - entry_pos
            if held < rule.min_hold or trough > rule.trough_limit:
                continue
            if current >= rule.recovery_target:
                signal_pos = pos
                execute_pos = pos + 1
                break
        if execute_pos is not None and execute_pos <= exit_pos:
            original_exit = float(trade["exit_close"])
            new_exit = float(closes[execute_pos])
            item["exit_date"] = pd.Timestamp(dates[execute_pos])
            item["exit_close"] = new_exit
            item["ret"] = (1.0 + float(trade["ret"])) * new_exit / original_exit - 1.0
            item["deep_recovery_triggered"] = True
            item["deep_recovery_signal_date"] = pd.Timestamp(dates[signal_pos])
            item["deep_recovery_trough"] = trough
        rows.append(item)
    return pd.DataFrame(rows).sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def score(stat: dict) -> float:
    return math.log(max(stat["final_1000"], 1.0) / 1000.0) + 0.65 * stat["win_rate"] + 0.80 * stat["max_drawdown"]


def neighbor_count(rule: Rule, eligible: set[str]) -> int:
    coordinates = (
        TREND_LIMITS.index(rule.trend_limit),
        MIN_HOLDS.index(rule.min_hold),
        TROUGH_LIMITS.index(rule.trough_limit),
        RECOVERY_TARGETS.index(rule.recovery_target),
    )
    count = 0
    for other in RULES:
        if other.key not in eligible or other.scope != rule.scope:
            continue
        other_coordinates = (
            TREND_LIMITS.index(other.trend_limit),
            MIN_HOLDS.index(other.min_hold),
            TROUGH_LIMITS.index(other.trough_limit),
            RECOVERY_TARGETS.index(other.recovery_target),
        )
        if sum(abs(a - b) for a, b in zip(coordinates, other_coordinates)) <= 1:
            count += 1
    return count


def select_rule(baseline: pd.DataFrame, logs: dict[str, pd.DataFrame], year: int, window: int | None) -> tuple[Rule | None, dict]:
    min_year = int(baseline["original_year"].min())
    start = min_year if window is None else max(min_year, year - window)
    years = set(range(start, year))
    base = metrics(baseline, years)
    if base["trades"] < 15:
        return None, {"fallback": True, **base}
    rows = []
    for rule in RULES:
        stat = metrics(logs[rule.key], years)
        eligible = bool(
            stat["trades"] == base["trades"]
            and stat["final_1000"] > base["final_1000"] * 1.01
            and stat["win_rate"] > base["win_rate"] + 0.005
            and stat["max_drawdown"] >= base["max_drawdown"] - 0.01
        )
        rows.append({"rule": rule, "key": rule.key, "eligible": eligible, "score": score(stat), **stat})
    table = pd.DataFrame(rows)
    eligible = set(table.loc[table["eligible"], "key"])
    if not eligible:
        return None, {"fallback": True, **base}
    table["neighbors"] = table["rule"].map(lambda value: neighbor_count(value, eligible))
    stable = table[table["eligible"] & table["neighbors"].ge(2)].sort_values(
        ["score", "final_1000", "win_rate"], ascending=False
    )
    if stable.empty:
        return None, {"fallback": True, **base}
    winner = stable.iloc[0]
    return winner["rule"], winner.drop(labels=["rule"]).to_dict()


def rolling(baseline: pd.DataFrame, logs: dict[str, pd.DataFrame], window: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    min_year = int(baseline["original_year"].min())
    first_year = max(2019, min_year + 3)
    parts = [baseline[baseline["original_year"].lt(first_year)].copy()]
    choices = []
    for year in range(first_year, 2027):
        rule, audit = select_rule(baseline, logs, year, window)
        source = baseline if rule is None else logs[rule.key]
        parts.append(source[source["original_year"].eq(year)].copy())
        choices.append(
            {
                "year": year,
                "window": "all" if window is None else window,
                "train_start": min_year if window is None else max(min_year, year - window),
                "train_end": year - 1,
                "rule": "BASE_EXIT" if rule is None else rule.key,
                "train_score": audit.get("score"),
                "train_trades": audit.get("trades"),
                "neighbors": audit.get("neighbors", 0),
            }
        )
    result = pd.concat(parts, ignore_index=True, sort=False).sort_values(["entry_date", "symbol"])
    return result, pd.DataFrame(choices)


def run_line(name: str, baseline: pd.DataFrame, raw: pd.DataFrame) -> dict:
    logs = {rule.key: adjust(baseline, raw, rule) for rule in RULES}
    base_full = metrics(baseline)
    base_dev = metrics(baseline, DEV_META_YEARS)
    base_holdout = metrics(baseline, HOLDOUT_YEARS)
    rows = []
    generated = {}
    for window in WINDOWS:
        key = "all" if window is None else str(window)
        trades, choices = rolling(baseline, logs, window)
        generated[key] = (trades, choices)
        rows.append(
            {
                "window": key,
                **{f"full_{k}": v for k, v in metrics(trades).items()},
                **{f"dev_{k}": v for k, v in metrics(trades, DEV_META_YEARS).items()},
                **{f"holdout_{k}": v for k, v in metrics(trades, HOLDOUT_YEARS).items()},
            }
        )
    table = pd.DataFrame(rows)
    dev_ok = table[
        table["dev_final_1000"].gt(base_dev["final_1000"])
        & table["dev_win_rate"].gt(base_dev["win_rate"])
    ].copy()
    if dev_ok.empty:
        chosen = "all"
    else:
        dev_ok["score"] = (
            np.log(dev_ok["dev_final_1000"] / 1000.0)
            + 0.65 * dev_ok["dev_win_rate"]
            + 0.80 * dev_ok["dev_max_drawdown"]
        )
        chosen = str(dev_ok.sort_values("score", ascending=False).iloc[0]["window"])
    winner, choices = generated[chosen]
    full = metrics(winner)
    holdout = metrics(winner, HOLDOUT_YEARS)
    robust = int(
        (
            table["holdout_final_1000"].gt(base_holdout["final_1000"])
            & table["holdout_win_rate"].gt(base_holdout["win_rate"])
            & table["holdout_max_drawdown"].ge(base_holdout["max_drawdown"])
        ).sum()
    )
    promoted = bool(
        full["final_1000"] > base_full["final_1000"]
        and full["win_rate"] > base_full["win_rate"]
        and holdout["final_1000"] > base_holdout["final_1000"]
        and holdout["win_rate"] > base_holdout["win_rate"]
        and holdout["max_drawdown"] >= base_holdout["max_drawdown"]
        and robust >= 2
    )
    table.to_csv(OUT / f"{name.lower()}_variants.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(OUT / f"{name.lower()}_choices.csv", index=False, encoding="utf-8-sig")
    winner.to_csv(OUT / f"{name.lower()}_winner_trades.csv", index=False, encoding="utf-8-sig")
    return {
        "line": name,
        "selected_window_on_development": chosen,
        "baseline_full": base_full,
        "winner_full": full,
        "baseline_holdout": base_holdout,
        "winner_holdout": holdout,
        "holdout_neighbor_windows_both_improved": robust,
        "promoted": promoted,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw, _ = fg.load_clean_raw()
    c = attach_pre_entry_ret60(fill_trade_prices(read_trades(C_TRADES), raw), raw)
    s = attach_pre_entry_ret60(fill_trade_prices(read_trades(S_TRADES), raw), raw)
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test": "C/S annual past-only deep-downtrend recovery exit",
        "execution": "recovery signal after close, execute next trading-day close",
        "C": run_line("C", c, raw),
        "S": run_line("S", s, raw),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
