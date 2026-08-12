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


OUT = ROOT / "fit" / "trailing_profit_exit_test"
C_TRADES = ROOT / "fit" / "hybrid_formal" / "final_trades.csv"
S_TRADES = PROJECT / "S" / "fit" / "selector" / "final_trades.csv"
DEV_META_YEARS = {2019, 2020, 2021, 2022, 2023}
HOLDOUT_YEARS = {2024, 2025, 2026}
WINDOWS: tuple[int | None, ...] = (3, 5, None)
MIN_HOLDS = (5, 10, 20)
ARMS = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12)
TRAILS = (0.01, 0.02, 0.03, 0.05, 0.07)


@dataclass(frozen=True)
class Rule:
    scope: str
    min_hold: int
    arm: float
    trail: float

    @property
    def key(self) -> str:
        return f"{self.scope}:h{self.min_hold}:arm{int(self.arm * 100)}:trail{int(self.trail * 100)}"


RULES = tuple(
    Rule(scope, hold, arm, trail)
    for scope in ("all", "main", "fallback")
    for hold in MIN_HOLDS
    for arm in ARMS
    for trail in TRAILS
)


def targeted(source: object, scope: str) -> bool:
    main = str(source) == "主策略"
    return scope == "all" or (scope == "main" and main) or (scope == "fallback" and not main)


def adjust(frame: pd.DataFrame, raw: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    calendars = {
        symbol: group.sort_values("date").reset_index(drop=True)
        for symbol, group in raw.groupby("symbol", sort=False)
    }
    rows = []
    for _, trade in frame.iterrows():
        item = trade.to_dict()
        item["trailing_rule"] = rule.key
        item["trailing_triggered"] = False
        if not targeted(trade.get("source"), rule.scope):
            rows.append(item)
            continue
        data = calendars.get(str(trade["symbol"]))
        if data is None:
            rows.append(item)
            continue
        dates = pd.DatetimeIndex(data["date"])
        entry_pos = int(dates.searchsorted(pd.Timestamp(trade["entry_date"])))
        exit_pos = int(dates.searchsorted(pd.Timestamp(trade["exit_date"])))
        if entry_pos >= len(data) or exit_pos >= len(data) or exit_pos <= entry_pos:
            rows.append(item)
            continue
        closes = pd.to_numeric(data["close"], errors="coerce").to_numpy(dtype=float)
        entry_price = float(trade["entry_close"])
        peak = -np.inf
        trigger = None
        for pos in range(entry_pos, exit_pos + 1):
            current = closes[pos] / entry_price - 1.0
            peak = max(peak, current)
            if pos - entry_pos < rule.min_hold:
                continue
            if peak >= rule.arm and peak - current >= rule.trail:
                if pos + 1 <= exit_pos:
                    trigger = pos + 1
                break
        if trigger is not None:
            original_exit = float(trade["exit_close"])
            new_exit = float(closes[trigger])
            item["exit_date"] = pd.Timestamp(dates[trigger])
            item["exit_close"] = new_exit
            item["ret"] = (1.0 + float(trade["ret"])) * new_exit / original_exit - 1.0
            item["trailing_triggered"] = True
            item["trailing_signal_date"] = pd.Timestamp(dates[trigger - 1])
            item["trailing_peak_return"] = peak
            item["trailing_current_return"] = closes[trigger - 1] / entry_price - 1.0
        rows.append(item)
    return pd.DataFrame(rows).sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def score(stat: dict) -> float:
    return math.log(max(stat["final_1000"], 1.0) / 1000.0) + 0.65 * stat["win_rate"] + 0.80 * stat["max_drawdown"]


def neighbors(rule: Rule, eligible: set[str]) -> int:
    hi = MIN_HOLDS.index(rule.min_hold)
    ai = ARMS.index(rule.arm)
    ti = TRAILS.index(rule.trail)
    count = 0
    for other in RULES:
        if other.key not in eligible or other.scope != rule.scope:
            continue
        distance = abs(MIN_HOLDS.index(other.min_hold) - hi) + abs(ARMS.index(other.arm) - ai) + abs(TRAILS.index(other.trail) - ti)
        if distance <= 1:
            count += 1
    return count


def choose(baseline: pd.DataFrame, logs: dict[str, pd.DataFrame], year: int, window: int | None) -> tuple[Rule | None, dict]:
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
            stat["trades"] >= 15
            and stat["final_1000"] > base["final_1000"] * 1.01
            and stat["win_rate"] > base["win_rate"] + 0.005
            and stat["max_drawdown"] >= base["max_drawdown"] - 0.01
        )
        rows.append({"rule": rule, "key": rule.key, "eligible": eligible, "score": score(stat), **stat})
    table = pd.DataFrame(rows)
    eligible = set(table.loc[table["eligible"], "key"])
    if not eligible:
        return None, {"fallback": True, **base}
    table["neighbors"] = table["rule"].map(lambda item: neighbors(item, eligible))
    stable = table[table["eligible"] & table["neighbors"].ge(2)].sort_values(["score", "final_1000", "win_rate"], ascending=False)
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
        rule, audit = choose(baseline, logs, year, window)
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
    return pd.concat(parts, ignore_index=True, sort=False).sort_values(["entry_date", "symbol"]), pd.DataFrame(choices)


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
    dev_ok = table[table["dev_final_1000"].gt(base_dev["final_1000"]) & table["dev_win_rate"].gt(base_dev["win_rate"])].copy()
    if dev_ok.empty:
        chosen = "all"
    else:
        dev_ok["score"] = np.log(dev_ok["dev_final_1000"] / 1000.0) + 0.65 * dev_ok["dev_win_rate"] + 0.80 * dev_ok["dev_max_drawdown"]
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
    c = fill_trade_prices(read_trades(C_TRADES), raw)
    s = fill_trade_prices(read_trades(S_TRADES), raw)
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test": "C/S annual past-only activated trailing-profit exit",
        "execution": "drawdown signal after close, execute next trading-day close",
        "C": run_line("C", c, raw),
        "S": run_line("S", s, raw),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
