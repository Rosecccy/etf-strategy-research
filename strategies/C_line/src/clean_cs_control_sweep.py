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
WORKSPACE = ROOT.parent
OUT = ROOT / "fit" / "clean_control_sweep"
DEV_YEARS = tuple(range(2019, 2024))
HOLDOUT_YEARS = (2024, 2025, 2026)
WINDOWS: tuple[int | None, ...] = (3, 5, None)
COMMISSION_RATE = 0.0003
MIN_COMMISSION = 5.0


@dataclass(frozen=True)
class EntryRule:
    feature: str
    direction: str
    threshold: float

    @property
    def key(self) -> str:
        return f"entry:{self.feature}:{self.direction}:{self.threshold:+.4f}"


@dataclass(frozen=True)
class ExitRule:
    feature: str
    threshold: float
    extension: int

    @property
    def key(self) -> str:
        return f"extend:{self.feature}:{self.threshold:+.4f}:d{self.extension}"


@dataclass(frozen=True)
class WeakExitRule:
    feature: str
    threshold: float
    min_hold: int

    @property
    def key(self) -> str:
        return f"weak_exit:{self.feature}:{self.threshold:+.4f}:h{self.min_hold}"


@dataclass(frozen=True)
class TrailRule:
    min_hold: int
    arm: float
    trail: float

    @property
    def key(self) -> str:
        return f"trail:h{self.min_hold}:arm{self.arm:.3f}:trail{self.trail:.3f}"


ENTRY_RULES = tuple(
    [EntryRule("ret5", "ge", value) for value in (-0.08, -0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05)]
    + [EntryRule("ret20", "ge", value) for value in (-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10)]
    + [EntryRule("ma60_gap", "ge", value) for value in (-0.20, -0.15, -0.10, -0.05, 0.0, 0.05)]
    + [EntryRule("drawdown60", "le", value) for value in (-0.02, -0.05, -0.10, -0.15, -0.20, -0.30)]
    + [EntryRule("drawdown60", "ge", value) for value in (-0.30, -0.25, -0.20, -0.18, -0.15, -0.12, -0.10, -0.08, -0.05)]
    + [EntryRule("vol20", "le", value) for value in (0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00)]
    + [EntryRule("rebound5", "ge", value) for value in (0.0, 0.01, 0.02, 0.03, 0.05)]
)

EXIT_RULES = tuple(
    [ExitRule("ret20", threshold, extension) for threshold in (-0.02, 0.0, 0.02, 0.05, 0.08, 0.12) for extension in (3, 5, 10, 20, 30, 40)]
    + [ExitRule("ma20_gap", threshold, extension) for threshold in (-0.02, 0.0, 0.02, 0.05, 0.08) for extension in (3, 5, 10, 20, 30, 40)]
)

WEAK_EXIT_RULES = tuple(
    [WeakExitRule(feature, threshold, hold)
     for feature in ("ret20", "ma20_gap")
     for threshold in (-0.15, -0.10, -0.07, -0.05, -0.03, 0.0)
     for hold in (5, 10, 20, 30)]
)

TRAIL_RULES = tuple(
    TrailRule(hold, arm, trail)
    for hold in (5, 10, 20, 30)
    for arm in (0.02, 0.03, 0.05, 0.08, 0.12, 0.20)
    for trail in (0.02, 0.03, 0.05, 0.07, 0.10)
)


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_raw(project: Path) -> pd.DataFrame:
    quality = read_json(project / "raw" / "quality.json")
    if not bool(quality.get("passed")):
        raise RuntimeError(f"Quality gate failed: {project.name}")
    pieces = []
    for symbol in quality["approved_symbols"]:
        path = project / "raw" / "etf" / f"{str(symbol).zfill(6)}.csv"
        if not path.exists() and project.name == "S":
            path = ROOT / "raw" / "etf" / f"{str(symbol).zfill(6)}.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame["symbol"] = str(symbol).zfill(6)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in ("close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        pieces.append(frame.dropna(subset=["date", "close"]))
    return pd.concat(pieces, ignore_index=True).sort_values(["symbol", "date"])


def read_trades(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    frame["exit_date"] = pd.to_datetime(frame["exit_date"], errors="coerce")
    frame["entry_close"] = pd.to_numeric(frame["entry_close"], errors="coerce")
    frame["exit_close"] = pd.to_numeric(frame["exit_close"], errors="coerce")
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    frame["year"] = frame["entry_date"].dt.year
    return frame.dropna(subset=["entry_date", "exit_date", "ret"]).sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def feature_panel(raw: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, group in raw.groupby("symbol", sort=False):
        data = group.sort_values("date").copy()
        close = data["close"]
        returns = close.pct_change()
        data["ret5"] = close.pct_change(5)
        data["ret20"] = close.pct_change(20)
        data["ma20_gap"] = close / close.rolling(20, min_periods=20).mean() - 1.0
        data["ma60_gap"] = close / close.rolling(60, min_periods=60).mean() - 1.0
        data["drawdown60"] = close / close.rolling(60, min_periods=60).max() - 1.0
        data["vol20"] = returns.rolling(20, min_periods=20).std() * np.sqrt(252)
        data["rebound5"] = close / close.rolling(5, min_periods=5).min() - 1.0
        pieces.append(data[["symbol", "date", "ret5", "ret20", "ma20_gap", "ma60_gap", "drawdown60", "vol20", "rebound5"]])
    return pd.concat(pieces, ignore_index=True).sort_values(["symbol", "date"])


def attach_entry_features(trades: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = {symbol: group.set_index("date").sort_index() for symbol, group in panel.groupby("symbol", sort=False)}
    feature_names = ["ret5", "ret20", "ma20_gap", "ma60_gap", "drawdown60", "vol20", "rebound5"]
    for _, trade in trades.iterrows():
        item = trade.to_dict()
        history = grouped[str(trade["symbol"])]
        prior = history[history.index < pd.Timestamp(trade["entry_date"])]
        state = prior.iloc[-1]
        item["entry_state_date"] = prior.index[-1]
        for feature in feature_names:
            item[feature] = state[feature]
        rows.append(item)
    return pd.DataFrame(rows).sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def apply_entry_rule(frame: pd.DataFrame, rule: EntryRule | None) -> pd.DataFrame:
    if rule is None:
        return frame.copy()
    values = pd.to_numeric(frame[rule.feature], errors="coerce")
    passed = values.ge(rule.threshold) if rule.direction == "ge" else values.le(rule.threshold)
    result = frame[passed.fillna(False)].copy()
    result["control_rule"] = rule.key
    return result


def extend_exits(frame: pd.DataFrame, raw: pd.DataFrame, rule: ExitRule | None) -> pd.DataFrame:
    if rule is None:
        return frame.copy()
    calendars = {symbol: group.sort_values("date").reset_index(drop=True) for symbol, group in raw.groupby("symbol", sort=False)}
    original = frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)
    rows = []
    for index, trade in original.iterrows():
        item = trade.to_dict()
        data = calendars[str(trade["symbol"])]
        dates = pd.DatetimeIndex(data["date"])
        closes = data["close"].to_numpy(dtype=float)
        exit_pos = int(dates.searchsorted(pd.Timestamp(trade["exit_date"]), side="left"))
        signal_pos = exit_pos - 1
        if signal_pos < 0 or exit_pos >= len(data):
            rows.append(item)
            continue
        signal_date = dates[signal_pos]
        feature_data = data[data["date"].le(signal_date)].copy()
        close = feature_data["close"]
        if rule.feature == "ret20":
            value = close.pct_change(20).iloc[-1]
        else:
            value = close.iloc[-1] / close.rolling(20, min_periods=20).mean().iloc[-1] - 1.0
        if pd.notna(value) and float(value) >= rule.threshold:
            target_pos = min(exit_pos + rule.extension, len(data) - 1)
            if index + 1 < len(original):
                next_entry = pd.Timestamp(original.iloc[index + 1]["entry_date"])
                cap_pos = int(dates.searchsorted(next_entry, side="left"))
                if cap_pos < len(dates):
                    target_pos = min(target_pos, cap_pos)
            if target_pos > exit_pos:
                item["exit_date"] = pd.Timestamp(dates[target_pos])
                item["exit_close"] = float(closes[target_pos])
                item["ret"] = float(closes[target_pos] / float(trade["entry_close"]) - 1.0)
                item["exit_extended"] = True
                item["exit_state_value"] = float(value)
        item["control_rule"] = rule.key
        rows.append(item)
    return enforce_single_position(pd.DataFrame(rows))


def early_weak_exits(frame: pd.DataFrame, raw: pd.DataFrame, rule: WeakExitRule | None) -> pd.DataFrame:
    if rule is None:
        return frame.copy()
    calendars = {symbol: group.sort_values("date").reset_index(drop=True) for symbol, group in raw.groupby("symbol", sort=False)}
    rows = []
    for _, trade in frame.iterrows():
        item = trade.to_dict()
        item["control_rule"] = rule.key
        item["weak_exit_triggered"] = False
        data = calendars.get(str(trade["symbol"]))
        if data is None:
            rows.append(item)
            continue
        dates = pd.DatetimeIndex(data["date"])
        closes = pd.to_numeric(data["close"], errors="coerce").to_numpy(dtype=float)
        entry_pos = int(dates.searchsorted(pd.Timestamp(trade["entry_date"]), side="left"))
        exit_pos = int(dates.searchsorted(pd.Timestamp(trade["exit_date"]), side="left"))
        trigger_pos = None
        start = entry_pos + rule.min_hold
        for signal_pos in range(start, max(start, exit_pos)):
            if signal_pos < 20 or signal_pos + 1 > exit_pos:
                continue
            if rule.feature == "ret20":
                value = closes[signal_pos] / closes[signal_pos - 20] - 1.0
            else:
                value = closes[signal_pos] / float(np.mean(closes[signal_pos - 19:signal_pos + 1])) - 1.0
            if np.isfinite(value) and value <= rule.threshold:
                trigger_pos = signal_pos + 1
                item["weak_exit_signal_value"] = float(value)
                break
        if trigger_pos is not None and trigger_pos < exit_pos:
            item["exit_date"] = pd.Timestamp(dates[trigger_pos])
            item["exit_close"] = float(closes[trigger_pos])
            item["ret"] = float(closes[trigger_pos] / float(trade["entry_close"]) - 1.0)
            item["weak_exit_triggered"] = True
            item["weak_exit_signal_date"] = pd.Timestamp(dates[trigger_pos - 1])
        rows.append(item)
    return enforce_single_position(pd.DataFrame(rows))


def trailing_exits(frame: pd.DataFrame, raw: pd.DataFrame, rule: TrailRule | None) -> pd.DataFrame:
    if rule is None:
        return frame.copy()
    calendars = {symbol: group.sort_values("date").reset_index(drop=True) for symbol, group in raw.groupby("symbol", sort=False)}
    rows = []
    for _, trade in frame.iterrows():
        item = trade.to_dict()
        item["control_rule"] = rule.key
        item["trail_triggered"] = False
        data = calendars.get(str(trade["symbol"]))
        if data is None:
            rows.append(item)
            continue
        dates = pd.DatetimeIndex(data["date"])
        closes = pd.to_numeric(data["close"], errors="coerce").to_numpy(dtype=float)
        entry_pos = int(dates.searchsorted(pd.Timestamp(trade["entry_date"]), side="left"))
        exit_pos = int(dates.searchsorted(pd.Timestamp(trade["exit_date"]), side="left"))
        entry_price = float(trade["entry_close"])
        peak = -np.inf
        trigger_pos = None
        for signal_pos in range(entry_pos, exit_pos):
            current_return = closes[signal_pos] / entry_price - 1.0
            peak = max(peak, current_return)
            if signal_pos - entry_pos < rule.min_hold:
                continue
            if peak >= rule.arm and peak - current_return >= rule.trail and signal_pos + 1 <= exit_pos:
                trigger_pos = signal_pos + 1
                item["trail_peak_return"] = float(peak)
                item["trail_signal_return"] = float(current_return)
                break
        if trigger_pos is not None and trigger_pos < exit_pos:
            item["exit_date"] = pd.Timestamp(dates[trigger_pos])
            item["exit_close"] = float(closes[trigger_pos])
            item["ret"] = float(closes[trigger_pos] / entry_price - 1.0)
            item["trail_triggered"] = True
            item["trail_signal_date"] = pd.Timestamp(dates[trigger_pos - 1])
        rows.append(item)
    return enforce_single_position(pd.DataFrame(rows))


def enforce_single_position(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    available = pd.Timestamp.min
    for _, trade in frame.sort_values(["entry_date", "symbol"]).iterrows():
        if pd.Timestamp(trade["entry_date"]) < available:
            continue
        rows.append(trade.to_dict())
        available = pd.Timestamp(trade["exit_date"])
    result = pd.DataFrame(rows)
    if len(result):
        result["year"] = pd.to_datetime(result["entry_date"]).dt.year
    return result


def fee(notional: float) -> float:
    return max(MIN_COMMISSION, abs(notional) * COMMISSION_RATE)


def metrics(frame: pd.DataFrame, years: tuple[int, ...] | None = None) -> dict:
    local = frame.copy()
    if years is None:
        years = tuple(range(int(frame["year"].min()), 2027))
    local = local[local["year"].isin(years)].sort_values(["entry_date", "symbol"])
    returns = pd.to_numeric(local["ret"], errors="coerce").dropna()
    equity = 1000.0 * (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0 if len(equity) else pd.Series(dtype=float)
    annual = local.groupby("year")["ret"].apply(lambda values: float(np.prod(1.0 + values.astype(float)) - 1.0)).reindex(years, fill_value=0.0)
    cash = 10_000.0
    for _, trade in local.iterrows():
        entry = float(trade["entry_close"])
        exit_ = float(trade["exit_close"])
        quantity = math.floor(cash / (entry * 100)) * 100
        while quantity >= 100 and quantity * entry + fee(quantity * entry) > cash:
            quantity -= 100
        if quantity < 100:
            continue
        buy = quantity * entry
        sell = quantity * exit_
        cash += sell - fee(sell) - buy - fee(buy)
    return {
        "trades": int(len(returns)),
        "gross_final_1000": float(equity.iloc[-1]) if len(equity) else 1000.0,
        "net_final_10000": float(cash),
        "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        "avg_return": float(returns.mean()) if len(returns) else 0.0,
        "avg_annual_return": float(annual.mean()),
        "positive_year_rate": float((annual > 0).mean()),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
    }


def score(stat: dict) -> float:
    return (
        math.log(max(float(stat["gross_final_1000"]), 1.0) / 1000.0)
        + 0.70 * float(stat["win_rate"])
        + 0.75 * float(stat["avg_annual_return"])
        + 0.80 * float(stat["max_drawdown"])
    )


def neighbor_count(rule, eligible: set[str], rules: tuple) -> int:
    if isinstance(rule, TrailRule):
        hold_values = sorted({item.min_hold for item in rules})
        arm_values = sorted({item.arm for item in rules})
        trail_values = sorted({item.trail for item in rules})
        hi = hold_values.index(rule.min_hold)
        ai = arm_values.index(rule.arm)
        ti = trail_values.index(rule.trail)
        return sum(
            item.key in eligible
            and abs(hold_values.index(item.min_hold) - hi)
            + abs(arm_values.index(item.arm) - ai)
            + abs(trail_values.index(item.trail) - ti) <= 1
            for item in rules
        )
    family = [item for item in rules if item.feature == rule.feature]
    if isinstance(rule, EntryRule):
        family = [item for item in family if item.direction == rule.direction]
        index = family.index(rule)
        return sum(item.key in eligible and abs(family.index(item) - index) <= 1 for item in family)
    if isinstance(rule, WeakExitRule):
        threshold_values = sorted({item.threshold for item in family})
        hold_values = sorted({item.min_hold for item in family})
        ti = threshold_values.index(rule.threshold)
        hi = hold_values.index(rule.min_hold)
        return sum(
            item.key in eligible
            and abs(threshold_values.index(item.threshold) - ti) + abs(hold_values.index(item.min_hold) - hi) <= 1
            for item in family
        )
    threshold_values = sorted({item.threshold for item in family})
    extension_values = sorted({item.extension for item in family})
    ti = threshold_values.index(rule.threshold)
    ei = extension_values.index(rule.extension)
    return sum(
        item.key in eligible
        and abs(threshold_values.index(item.threshold) - ti) + abs(extension_values.index(item.extension) - ei) <= 1
        for item in family
    )


def choose_rule(baseline: pd.DataFrame, variants: dict[str, pd.DataFrame], rules: tuple, year: int, window: int | None):
    first = int(baseline["year"].min())
    start = first if window is None else max(first, year - window)
    years = tuple(range(start, year))
    base = metrics(baseline, years)
    if base["trades"] < 7:
        return None, {"fallback": True, **base}
    rows = []
    for rule in rules:
        stat = metrics(variants[rule.key], years)
        retain = stat["trades"] / max(1, base["trades"])
        eligible = bool(
            stat["trades"] >= max(6, math.ceil(base["trades"] * 0.65))
            and retain >= 0.65
            and stat["gross_final_1000"] > base["gross_final_1000"] * 1.005
            and stat["win_rate"] >= base["win_rate"]
            and stat["avg_annual_return"] > base["avg_annual_return"]
            and stat["max_drawdown"] >= base["max_drawdown"] - 0.015
        )
        rows.append({"rule": rule, "key": rule.key, "eligible": eligible, "score": score(stat), **stat})
    table = pd.DataFrame(rows)
    eligible = set(table.loc[table["eligible"], "key"])
    if not eligible:
        return None, {"fallback": True, **base}
    table["neighbors"] = table["rule"].map(lambda item: neighbor_count(item, eligible, rules))
    stable = table[table["eligible"] & table["neighbors"].ge(2)].sort_values(
        ["score", "gross_final_1000", "win_rate"], ascending=False
    )
    if stable.empty:
        return None, {"fallback": True, **base}
    winner = stable.iloc[0]
    return winner["rule"], winner.drop(labels=["rule"]).to_dict()


def rolling(baseline: pd.DataFrame, variants: dict[str, pd.DataFrame], rules: tuple, window: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_year = max(2019, int(baseline["year"].min()) + 3)
    parts = [baseline[baseline["year"].lt(first_year)].copy()]
    choices = []
    for year in range(first_year, 2027):
        rule, audit = choose_rule(baseline, variants, rules, year, window)
        source = baseline if rule is None else variants[rule.key]
        parts.append(source[source["year"].eq(year)].copy())
        choices.append(
            {
                "year": year,
                "window": "all" if window is None else window,
                "train_end": year - 1,
                "rule": "BASE" if rule is None else rule.key,
                "train_score": audit.get("score"),
                "train_trades": audit.get("trades"),
                "neighbors": audit.get("neighbors", 0),
            }
        )
    return enforce_single_position(pd.concat(parts, ignore_index=True, sort=False)), pd.DataFrame(choices)


def run_module(line: str, module: str, baseline: pd.DataFrame, raw: pd.DataFrame, panel: pd.DataFrame) -> dict:
    if module == "entry":
        enriched = attach_entry_features(baseline, panel)
        base = enriched
        rules = ENTRY_RULES
        variants = {rule.key: apply_entry_rule(enriched, rule) for rule in rules}
    elif module == "extension":
        base = baseline
        rules = EXIT_RULES
        variants = {rule.key: extend_exits(baseline, raw, rule) for rule in rules}
    elif module == "weak_exit":
        base = baseline
        rules = WEAK_EXIT_RULES
        variants = {rule.key: early_weak_exits(baseline, raw, rule) for rule in rules}
    else:
        base = baseline
        rules = TRAIL_RULES
        variants = {rule.key: trailing_exits(baseline, raw, rule) for rule in rules}
    base_full = metrics(base)
    base_dev = metrics(base, DEV_YEARS)
    base_holdout = metrics(base, HOLDOUT_YEARS)
    rows = []
    generated = {}
    for window in WINDOWS:
        key = "all" if window is None else str(window)
        trades, choices = rolling(base, variants, rules, window)
        generated[key] = (trades, choices)
        rows.append(
            {
                "window": key,
                **{f"full_{k}": v for k, v in metrics(trades).items()},
                **{f"dev_{k}": v for k, v in metrics(trades, DEV_YEARS).items()},
                **{f"holdout_{k}": v for k, v in metrics(trades, HOLDOUT_YEARS).items()},
            }
        )
    table = pd.DataFrame(rows)
    dev_ok = table[
        table["dev_gross_final_1000"].gt(base_dev["gross_final_1000"])
        & table["dev_win_rate"].ge(base_dev["win_rate"])
        & table["dev_avg_annual_return"].gt(base_dev["avg_annual_return"])
    ].copy()
    if dev_ok.empty:
        chosen = "all"
    else:
        dev_ok["selection_score"] = (
            np.log(dev_ok["dev_gross_final_1000"] / 1000.0)
            + 0.70 * dev_ok["dev_win_rate"]
            + 0.75 * dev_ok["dev_avg_annual_return"]
            + 0.80 * dev_ok["dev_max_drawdown"]
        )
        chosen = str(dev_ok.sort_values("selection_score", ascending=False).iloc[0]["window"])
    winner, choices = generated[chosen]
    full = metrics(winner)
    holdout = metrics(winner, HOLDOUT_YEARS)
    robust = int(
        (
            table["holdout_gross_final_1000"].gt(base_holdout["gross_final_1000"])
            & table["holdout_win_rate"].ge(base_holdout["win_rate"])
            & table["holdout_avg_annual_return"].gt(base_holdout["avg_annual_return"])
            & table["holdout_max_drawdown"].ge(base_holdout["max_drawdown"] - 0.005)
        ).sum()
    )
    promoted = bool(
        full["net_final_10000"] > base_full["net_final_10000"]
        and full["win_rate"] >= base_full["win_rate"]
        and full["avg_annual_return"] > base_full["avg_annual_return"]
        and holdout["gross_final_1000"] > base_holdout["gross_final_1000"]
        and holdout["win_rate"] >= base_holdout["win_rate"]
        and holdout["avg_annual_return"] > base_holdout["avg_annual_return"]
        and holdout["max_drawdown"] >= base_holdout["max_drawdown"] - 0.005
        and robust >= 2
    )
    prefix = f"{line.lower()}_{module}"
    table.to_csv(OUT / f"{prefix}_windows.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(OUT / f"{prefix}_choices.csv", index=False, encoding="utf-8-sig")
    winner.to_csv(OUT / f"{prefix}_trades.csv", index=False, encoding="utf-8-sig")
    return {
        "line": line,
        "module": module,
        "selected_window": chosen,
        "baseline_full": base_full,
        "winner_full": full,
        "baseline_holdout": base_holdout,
        "winner_holdout": holdout,
        "holdout_neighbor_windows_improved": robust,
        "promoted": promoted,
    }


def main() -> None:
    configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    c_raw = load_raw(ROOT)
    s_raw = load_raw(WORKSPACE / "S")
    c = read_trades(ROOT / "fit" / "formal_clean" / "final_trades.csv")
    s = read_trades(WORKSPACE / "S" / "fit" / "selector" / "formal_execution_trades.csv")
    requested = set(sys.argv[1:]) or {"entry", "extension", "weak_exit", "trailing"}
    results = []
    for line, trades, raw in (("C", c, c_raw), ("S", s, s_raw)):
        panel = feature_panel(raw)
        for module in ("entry", "extension", "weak_exit", "trailing"):
            if module in requested:
                results.append(run_module(line, module, trades, raw, panel))
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_quality": "C and S quality manifests must pass",
        "execution": "signals use only prior-close state; trades execute at next close; fees and 100-share lots included",
        "tests": results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
