from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
D_SRC = PROJECT / "D" / "src"
if str(D_SRC) not in sys.path:
    sys.path.insert(0, str(D_SRC))

import fear_greed_oos as fg  # noqa: E402


OUT = ROOT / "fit" / "stale_recovery_exit_overlay_test"
C_TRADES = ROOT / "fit" / "hybrid_formal" / "final_trades.csv"
S_TRADES = PROJECT / "S" / "fit" / "selector" / "final_trades.csv"
DEV_META_YEARS = {2019, 2020, 2021, 2022, 2023}
HOLDOUT_YEARS = {2024, 2025, 2026}
WINDOWS: tuple[int | None, ...] = (3, 5, None)
MIN_HOLDS = (20, 30, 40, 50, 60)
PEAK_CAPS = (0.01, 0.02, 0.03, 0.05)
RECOVERY_TARGETS = (0.0, 0.002, 0.005, 0.01)


@dataclass(frozen=True)
class Rule:
    scope: str
    min_hold: int
    peak_cap: float
    recovery_target: float

    @property
    def key(self) -> str:
        return (
            f"{self.scope}:h{self.min_hold}:cap{int(self.peak_cap * 1000)}:"
            f"target{int(self.recovery_target * 1000)}"
        )


RULES = tuple(
    Rule(scope, min_hold, peak_cap, target)
    for scope in ("all", "main", "fallback")
    for min_hold in MIN_HOLDS
    for peak_cap in PEAK_CAPS
    for target in RECOVERY_TARGETS
)


def read_trades(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    for column in ("entry_date", "exit_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in ("ret", "entry_close", "exit_close"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame["original_year"] = frame["entry_date"].dt.year
    return frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def fill_trade_prices(frame: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    price_map = raw.set_index(["symbol", "date"])["close"]
    entry_values = []
    exit_values = []
    for _, trade in result.iterrows():
        entry = pd.to_numeric(pd.Series([trade.get("entry_close")]), errors="coerce").iloc[0]
        exit_value = pd.to_numeric(pd.Series([trade.get("exit_close")]), errors="coerce").iloc[0]
        if not np.isfinite(entry):
            entry = price_map.get((str(trade["symbol"]), pd.Timestamp(trade["entry_date"])), np.nan)
        if not np.isfinite(exit_value):
            exit_value = price_map.get((str(trade["symbol"]), pd.Timestamp(trade["exit_date"])), np.nan)
        entry_values.append(float(entry) if pd.notna(entry) else np.nan)
        exit_values.append(float(exit_value) if pd.notna(exit_value) else np.nan)
    result["entry_close"] = entry_values
    result["exit_close"] = exit_values
    return result


def targeted(source: object, scope: str) -> bool:
    main = str(source) == "主策略"
    return scope == "all" or (scope == "main" and main) or (scope == "fallback" and not main)


def adjust_trades(frame: pd.DataFrame, raw: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    calendars = {
        symbol: group.sort_values("date").reset_index(drop=True)
        for symbol, group in raw.groupby("symbol", sort=False)
    }
    rows = []
    for _, trade in frame.iterrows():
        item = trade.to_dict()
        item["stale_exit_rule"] = rule.key
        item["stale_exit_triggered"] = False
        if not targeted(trade.get("source"), rule.scope):
            rows.append(item)
            continue
        data = calendars.get(str(trade["symbol"]))
        if data is None or data.empty:
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
        peak = -np.inf
        trigger = None
        for pos in range(entry_pos, exit_pos + 1):
            current_return = closes[pos] / entry_price - 1.0
            peak = max(peak, current_return)
            if pos - entry_pos < rule.min_hold:
                continue
            if peak < rule.peak_cap and current_return >= rule.recovery_target:
                if pos + 1 <= exit_pos:
                    trigger = pos + 1
                break
        if trigger is not None:
            original_exit = float(trade["exit_close"])
            new_exit = float(closes[trigger])
            item["exit_date"] = pd.Timestamp(dates[trigger])
            item["exit_close"] = new_exit
            item["ret"] = (1.0 + float(trade["ret"])) * new_exit / original_exit - 1.0
            item["stale_exit_triggered"] = True
            item["stale_signal_date"] = pd.Timestamp(dates[trigger - 1])
            item["stale_signal_peak_return"] = peak
            item["stale_signal_current_return"] = closes[trigger - 1] / entry_price - 1.0
        rows.append(item)
    return pd.DataFrame(rows).sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def metrics(frame: pd.DataFrame, years: set[int] | None = None) -> dict[str, float | int]:
    local = frame if years is None else frame[frame["original_year"].isin(years)]
    returns = pd.to_numeric(local.sort_values(["entry_date", "symbol"])["ret"], errors="coerce").dropna()
    if returns.empty:
        return {"trades": 0, "final_1000": 1000.0, "win_rate": 0.0, "max_drawdown": 0.0}
    equity = 1000.0 * (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "trades": int(len(returns)),
        "final_1000": float(equity.iloc[-1]),
        "win_rate": float((returns > 0).mean()),
        "max_drawdown": float(drawdown.min()),
    }


def score(stat: dict) -> float:
    return (
        math.log(max(float(stat["final_1000"]), 1.0) / 1000.0)
        + 0.65 * float(stat["win_rate"])
        + 0.80 * float(stat["max_drawdown"])
    )


def neighbor_count(rule: Rule, eligible: set[str]) -> int:
    hi = MIN_HOLDS.index(rule.min_hold)
    ci = PEAK_CAPS.index(rule.peak_cap)
    ti = RECOVERY_TARGETS.index(rule.recovery_target)
    count = 0
    for other in RULES:
        if other.key not in eligible or other.scope != rule.scope:
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
    baseline: pd.DataFrame,
    logs: dict[str, pd.DataFrame],
    year: int,
    window: int | None,
) -> tuple[Rule | None, dict]:
    min_year = int(baseline["original_year"].min())
    start = min_year if window is None else max(min_year, year - window)
    years = set(range(start, year))
    base = metrics(baseline, years)
    if base["trades"] < 12:
        return None, {"fallback": True, **base}
    rows = []
    for rule in RULES:
        stat = metrics(logs[rule.key], years)
        eligible = bool(
            stat["trades"] >= max(12, math.ceil(base["trades"] * 0.80))
            and stat["final_1000"] > base["final_1000"] * 1.01
            and stat["win_rate"] > base["win_rate"] + 0.005
            and stat["max_drawdown"] >= base["max_drawdown"] - 0.01
        )
        rows.append({"rule": rule, "key": rule.key, "eligible": eligible, "score": score(stat), **stat})
    table = pd.DataFrame(rows)
    eligible = set(table.loc[table["eligible"], "key"])
    if not eligible:
        return None, {"fallback": True, **base}
    table["neighbors"] = table["rule"].map(lambda item: neighbor_count(item, eligible))
    stable = table[table["eligible"] & table["neighbors"].ge(2)].sort_values(
        ["score", "final_1000", "win_rate"], ascending=False
    )
    if stable.empty:
        return None, {"fallback": True, **base}
    winner = stable.iloc[0]
    return winner["rule"], winner.drop(labels=["rule"]).to_dict()


def rolling(
    baseline: pd.DataFrame,
    logs: dict[str, pd.DataFrame],
    window: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    min_year = int(baseline["original_year"].min())
    first_year = max(2019, min_year + 3)
    parts = [baseline[baseline["original_year"].lt(first_year)].copy()]
    choices = []
    for year in range(first_year, 2027):
        rule, audit = choose_rule(baseline, logs, year, window)
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
    logs = {rule.key: adjust_trades(baseline, raw, rule) for rule in RULES}
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
        dev_ok["score"] = np.log(dev_ok["dev_final_1000"] / 1000.0) + 0.65 * dev_ok["dev_win_rate"] + 0.80 * dev_ok["dev_max_drawdown"]
        chosen = str(dev_ok.sort_values("score", ascending=False).iloc[0]["window"])
    winner, choices = generated[chosen]
    win_full = metrics(winner)
    win_holdout = metrics(winner, HOLDOUT_YEARS)
    robust = int(
        (
            table["holdout_final_1000"].gt(base_holdout["final_1000"])
            & table["holdout_win_rate"].gt(base_holdout["win_rate"])
            & table["holdout_max_drawdown"].ge(base_holdout["max_drawdown"])
        ).sum()
    )
    promoted = bool(
        win_full["final_1000"] > base_full["final_1000"]
        and win_full["win_rate"] > base_full["win_rate"]
        and win_holdout["final_1000"] > base_holdout["final_1000"]
        and win_holdout["win_rate"] > base_holdout["win_rate"]
        and win_holdout["max_drawdown"] >= base_holdout["max_drawdown"]
        and robust >= 2
    )
    table.to_csv(OUT / f"{name.lower()}_variants.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(OUT / f"{name.lower()}_choices.csv", index=False, encoding="utf-8-sig")
    winner.to_csv(OUT / f"{name.lower()}_winner_trades.csv", index=False, encoding="utf-8-sig")
    return {
        "line": name,
        "selected_window_on_development": chosen,
        "baseline_full": base_full,
        "winner_full": win_full,
        "baseline_holdout": base_holdout,
        "winner_holdout": win_holdout,
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
        "test": "C/S annual past-only stale-position recovery exit overlay",
        "execution": "exit signal after close, execute next trading-day close",
        "C": run_line("C", c, raw),
        "S": run_line("S", s, raw),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
