from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import fear_greed_oos as fg
from control_panic_age_test import portfolio_metrics


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "clean_control_sweep"
CANDIDATES = ROOT / "out" / "formal_clean" / "candidate_trades.csv"
WINDOWS: tuple[int | None, ...] = (3, 5, None)
DEV_YEARS = tuple(range(2019, 2024))
HOLDOUT_YEARS = (2024, 2025, 2026)


@dataclass(frozen=True)
class GateRule:
    feature: str
    direction: str
    threshold: float

    @property
    def key(self) -> str:
        return f"gate:{self.feature}:{self.direction}:{self.threshold:+.4f}"


@dataclass(frozen=True)
class RankRule:
    feature: str
    weight: float

    @property
    def key(self) -> str:
        return f"rank:{self.feature}:w{self.weight:.2f}"


@dataclass(frozen=True)
class ExitRule:
    feature: str
    threshold: float
    extension: int

    @property
    def key(self) -> str:
        return f"extend:{self.feature}:{self.threshold:+.4f}:d{self.extension}"


GATE_RULES = tuple(
    [GateRule("drawdown60", "ge", x) for x in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15)]
    + [GateRule("ret5", "le", x) for x in (-0.10, -0.07, -0.05, -0.03, -0.02, -0.01, 0.0, 0.02)]
    + [GateRule("ret20", "le", x) for x in (-0.20, -0.15, -0.10, -0.07, -0.05, -0.03, 0.0, 0.05)]
    + [GateRule("ma20_gap", "le", x) for x in (-0.10, -0.07, -0.05, -0.03, -0.01, 0.0, 0.02)]
)

RANK_RULES = tuple(
    RankRule(feature, weight)
    for feature in ("depth_score", "dip_score", "rebound_score")
    for weight in (0.10, 0.20, 0.30, 0.40, 0.50)
)

EXIT_RULES = tuple(
    ExitRule(feature, threshold, extension)
    for feature in ("ret20", "ma20_gap")
    for threshold in (-0.02, 0.0, 0.02, 0.05, 0.08, 0.12)
    for extension in (3, 5, 10, 20, 30)
)


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def feature_panel(raw: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for symbol, group in raw.groupby("symbol", sort=False):
        data = group.sort_values("date").copy()
        close = pd.to_numeric(data["close"], errors="coerce")
        data["ret1"] = close.pct_change()
        data["ret5"] = close.pct_change(5)
        data["ret20"] = close.pct_change(20)
        data["ma20_gap"] = close / close.rolling(20, min_periods=20).mean() - 1.0
        data["drawdown60"] = 1.0 - close / close.rolling(60, min_periods=60).max()
        data["rebound5"] = close / close.rolling(5, min_periods=5).min() - 1.0
        parts.append(data[["symbol", "date", "ret1", "ret5", "ret20", "ma20_gap", "drawdown60", "rebound5"]])
    return pd.concat(parts, ignore_index=True)


def attach_features(candidates: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    local = candidates.copy()
    local["symbol"] = local["symbol"].astype(str).str.zfill(6)
    local["signal_date"] = pd.to_datetime(local["signal_date"], errors="coerce")
    local["entry_date"] = pd.to_datetime(local["entry_date"], errors="coerce")
    local["exit_date"] = pd.to_datetime(local["exit_date"], errors="coerce")
    merged = local.merge(
        panel,
        left_on=["symbol", "signal_date"],
        right_on=["symbol", "date"],
        how="left",
    ).drop(columns=["date"], errors="ignore")
    merged["depth_score"] = merged.groupby("signal_date")["drawdown60"].rank(pct=True)
    merged["dip_score"] = merged.groupby("signal_date")["ret5"].rank(pct=True, ascending=False)
    merged["rebound_score"] = merged.groupby("signal_date")["ret1"].rank(pct=True)
    return merged.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def apply_gate(frame: pd.DataFrame, rule: GateRule) -> pd.DataFrame:
    values = pd.to_numeric(frame[rule.feature], errors="coerce")
    passed = values.ge(rule.threshold) if rule.direction == "ge" else values.le(rule.threshold)
    result = frame[passed.fillna(False)].copy()
    result["control_rule"] = rule.key
    return result


def apply_rank(frame: pd.DataFrame, rule: RankRule) -> pd.DataFrame:
    result = frame.copy()
    feature = pd.to_numeric(result[rule.feature], errors="coerce").fillna(0.5).clip(0, 1)
    current = pd.to_numeric(result["buy_quality"], errors="coerce").fillna(50.0)
    result["buy_quality"] = (1.0 - rule.weight) * current + rule.weight * feature * 100.0
    result["control_rule"] = rule.key
    return result


def exit_projection(frame: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    calendars = {symbol: group.sort_values("date").reset_index(drop=True) for symbol, group in raw.groupby("symbol", sort=False)}
    rows = []
    for _, trade in frame.iterrows():
        item = trade.to_dict()
        if str(trade.get("status", "closed")) != "closed" or pd.isna(trade.get("exit_date")):
            rows.append(item)
            continue
        data = calendars.get(str(trade["symbol"]))
        if data is None:
            rows.append(item)
            continue
        dates = pd.DatetimeIndex(data["date"])
        closes = pd.to_numeric(data["close"], errors="coerce").to_numpy(dtype=float)
        exit_pos = int(dates.searchsorted(pd.Timestamp(trade["exit_date"]), side="left"))
        signal_pos = exit_pos - 1
        if signal_pos < 20 or exit_pos >= len(data):
            rows.append(item)
            continue
        item["exit_ret20_state"] = float(closes[signal_pos] / closes[signal_pos - 20] - 1.0)
        item["exit_ma20_gap_state"] = float(closes[signal_pos] / np.mean(closes[signal_pos - 19:signal_pos + 1]) - 1.0)
        item["strong_exit_signal_date"] = pd.Timestamp(dates[signal_pos])
        for extension in sorted({rule.extension for rule in EXIT_RULES}):
            target = min(exit_pos + extension, len(data) - 1)
            item[f"future_date_{extension}"] = pd.Timestamp(dates[target])
            item[f"future_close_{extension}"] = float(closes[target])
        rows.append(item)
    return pd.DataFrame(rows).sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def extend_exits(frame: pd.DataFrame, rule: ExitRule) -> pd.DataFrame:
    result = frame.copy()
    state_column = f"exit_{rule.feature}_state"
    state = pd.to_numeric(result[state_column], errors="coerce")
    future_close = pd.to_numeric(result[f"future_close_{rule.extension}"], errors="coerce")
    mask = state.ge(rule.threshold) & future_close.notna() & result["status"].eq("closed")
    result["control_rule"] = rule.key
    result["strong_exit_extended"] = mask
    result.loc[mask, "exit_date"] = pd.to_datetime(result.loc[mask, f"future_date_{rule.extension}"])
    result.loc[mask, "exit_close"] = future_close[mask]
    result.loc[mask, "return_rate"] = future_close[mask] / pd.to_numeric(result.loc[mask, "entry_close"], errors="coerce") - 1.0
    result.loc[mask, "pnl_cny"] = result.loc[mask, "return_rate"] * 100.0
    result.loc[mask, "strong_exit_state"] = state[mask]
    return result.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def annual_stats(log: pd.DataFrame, years: tuple[int, ...]) -> tuple[float, float]:
    if log.empty:
        return 0.0, 0.0
    local = log.copy()
    local["entry_date"] = pd.to_datetime(local["entry_date"], errors="coerce")
    local["net_return"] = pd.to_numeric(local["net_return"], errors="coerce")
    closed = local[local["status_portfolio"].eq("closed") & local["net_return"].notna()].copy()
    values = []
    for year in years:
        ret = closed.loc[closed["entry_date"].dt.year.eq(year), "net_return"].astype(float)
        values.append(float(np.prod(1.0 + ret) - 1.0) if len(ret) else 0.0)
    series = pd.Series(values, dtype=float)
    return float(series.mean()), float((series > 0).mean())


def evaluate(frame: pd.DataFrame, raw: pd.DataFrame, pool: pd.DataFrame, years: tuple[int, ...]) -> tuple[pd.DataFrame, dict]:
    local = frame[frame["test_year"].isin(years)].copy()
    if local.empty:
        return pd.DataFrame(), {
            "final_value": 10000.0,
            "closed_trades": 0,
            "win_rate": 0.0,
            "avg_annual_return": 0.0,
            "positive_year_rate": 0.0,
            "max_drawdown": 0.0,
        }
    log, stat = portfolio_metrics(local, raw, pool)
    avg_annual, positive = annual_stats(log, years)
    stat["avg_annual_return"] = avg_annual
    stat["positive_year_rate"] = positive
    stat["win_rate"] = float(np.nan_to_num(stat.get("win_rate"), nan=0.0))
    return log, stat


def score(stat: dict) -> float:
    return (
        math.log(max(float(stat["final_value"]), 1.0) / 10000.0)
        + 0.70 * float(stat["win_rate"])
        + 0.75 * float(stat["avg_annual_return"])
        + 0.80 * float(stat["max_drawdown"])
    )


def neighbor_count(rule, eligible: set[str], rules: tuple) -> int:
    family = [item for item in rules if item.feature == rule.feature]
    if isinstance(rule, GateRule):
        family = [item for item in family if item.direction == rule.direction]
        values = sorted(item.threshold for item in family)
        index = values.index(rule.threshold)
        return sum(item.key in eligible and abs(values.index(item.threshold) - index) <= 1 for item in family)
    if isinstance(rule, RankRule):
        values = sorted(item.weight for item in family)
        index = values.index(rule.weight)
        return sum(item.key in eligible and abs(values.index(item.weight) - index) <= 1 for item in family)
    threshold_values = sorted({item.threshold for item in family})
    extension_values = sorted({item.extension for item in family})
    ti = threshold_values.index(rule.threshold)
    ei = extension_values.index(rule.extension)
    return sum(
        item.key in eligible
        and abs(threshold_values.index(item.threshold) - ti)
        + abs(extension_values.index(item.extension) - ei) <= 1
        for item in family
    )


def training_slice(frame: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    cutoff = pd.Timestamp(year=end + 1, month=1, day=1)
    return frame[frame["test_year"].between(start, end) & frame["exit_date"].lt(cutoff)].copy()


def choose_rule(
    baseline: pd.DataFrame,
    variants: dict[str, pd.DataFrame],
    rules: tuple,
    year: int,
    window: int | None,
    raw: pd.DataFrame,
    pool: pd.DataFrame,
):
    first = int(baseline["test_year"].min())
    start = first if window is None else max(first, year - window)
    years = tuple(range(start, year))
    base_frame = training_slice(baseline, start, year - 1)
    _, base = evaluate(base_frame, raw, pool, years)
    if base["closed_trades"] < 7:
        return None, {"fallback": True, **base}
    rows = []
    for rule in rules:
        variant = training_slice(variants[rule.key], start, year - 1)
        _, stat = evaluate(variant, raw, pool, years)
        eligible = bool(
            stat["closed_trades"] >= max(6, math.ceil(base["closed_trades"] * 0.60))
            and stat["final_value"] > base["final_value"] * 1.005
            and stat["win_rate"] >= base["win_rate"]
            and stat["avg_annual_return"] > base["avg_annual_return"]
            and stat["max_drawdown"] >= base["max_drawdown"] - 0.02
        )
        rows.append({"rule": rule, "key": rule.key, "eligible": eligible, "score": score(stat), **stat})
    table = pd.DataFrame(rows)
    eligible = set(table.loc[table["eligible"], "key"])
    if not eligible:
        return None, {"fallback": True, **base}
    table["neighbors"] = table["rule"].map(lambda item: neighbor_count(item, eligible, rules))
    stable = table[table["eligible"] & table["neighbors"].ge(2)].sort_values(
        ["score", "final_value", "win_rate"], ascending=False
    )
    if stable.empty:
        return None, {"fallback": True, **base}
    winner = stable.iloc[0]
    return winner["rule"], winner.drop(labels=["rule"]).to_dict()


def rolling(
    baseline: pd.DataFrame,
    variants: dict[str, pd.DataFrame],
    rules: tuple,
    window: int | None,
    raw: pd.DataFrame,
    pool: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_year = max(2022, int(baseline["test_year"].min()) + 3)
    parts = [baseline[baseline["test_year"].lt(first_year)].copy()]
    choices = []
    for year in range(first_year, 2027):
        rule, audit = choose_rule(baseline, variants, rules, year, window, raw, pool)
        source = baseline if rule is None else variants[rule.key]
        parts.append(source[source["test_year"].eq(year)].copy())
        choices.append({
            "year": year,
            "window": "all" if window is None else window,
            "rule": "BASE" if rule is None else rule.key,
            "train_score": audit.get("score"),
            "train_trades": audit.get("closed_trades"),
            "neighbors": audit.get("neighbors", 0),
        })
    return pd.concat(parts, ignore_index=True, sort=False), pd.DataFrame(choices)


def run_module(name: str, base: pd.DataFrame, variants: dict[str, pd.DataFrame], rules: tuple, raw: pd.DataFrame, pool: pd.DataFrame) -> dict:
    _, base_full = evaluate(base, raw, pool, tuple(range(2019, 2027)))
    _, base_dev = evaluate(base, raw, pool, DEV_YEARS)
    _, base_holdout = evaluate(base, raw, pool, HOLDOUT_YEARS)
    rows = []
    generated = {}
    for window in WINDOWS:
        key = "all" if window is None else str(window)
        candidates, choices = rolling(base, variants, rules, window, raw, pool)
        full_log, full = evaluate(candidates, raw, pool, tuple(range(2019, 2027)))
        _, dev = evaluate(candidates, raw, pool, DEV_YEARS)
        _, holdout = evaluate(candidates, raw, pool, HOLDOUT_YEARS)
        generated[key] = (candidates, choices, full_log, full, holdout)
        rows.append({
            "window": key,
            **{f"full_{k}": v for k, v in full.items()},
            **{f"dev_{k}": v for k, v in dev.items()},
            **{f"holdout_{k}": v for k, v in holdout.items()},
        })
    table = pd.DataFrame(rows)
    eligible = table[
        table["dev_final_value"].gt(base_dev["final_value"])
        & table["dev_win_rate"].ge(base_dev["win_rate"])
        & table["dev_avg_annual_return"].gt(base_dev["avg_annual_return"])
    ].copy()
    if eligible.empty:
        chosen = "all"
    else:
        eligible["selection_score"] = (
            np.log(eligible["dev_final_value"] / 10000.0)
            + 0.70 * eligible["dev_win_rate"]
            + 0.75 * eligible["dev_avg_annual_return"]
            + 0.80 * eligible["dev_max_drawdown"]
        )
        chosen = str(eligible.sort_values("selection_score", ascending=False).iloc[0]["window"])
    candidates, choices, full_log, full, holdout = generated[chosen]
    robust = int((
        table["holdout_final_value"].gt(base_holdout["final_value"])
        & table["holdout_win_rate"].ge(base_holdout["win_rate"])
        & table["holdout_avg_annual_return"].gt(base_holdout["avg_annual_return"])
        & table["holdout_max_drawdown"].ge(base_holdout["max_drawdown"] - 0.005)
    ).sum())
    promoted = bool(
        full["final_value"] > base_full["final_value"]
        and full["win_rate"] >= base_full["win_rate"]
        and full["avg_annual_return"] > base_full["avg_annual_return"]
        and holdout["final_value"] > base_holdout["final_value"]
        and holdout["win_rate"] >= base_holdout["win_rate"]
        and holdout["avg_annual_return"] > base_holdout["avg_annual_return"]
        and holdout["max_drawdown"] >= base_holdout["max_drawdown"] - 0.005
        and robust >= 2
    )
    table.to_csv(OUT / f"{name}_windows.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(OUT / f"{name}_choices.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(OUT / f"{name}_candidates.csv", index=False, encoding="utf-8-sig")
    full_log.to_csv(OUT / f"{name}_trades.csv", index=False, encoding="utf-8-sig")
    return {
        "module": name,
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
    quality = json.loads((ROOT.parent / "C" / "raw" / "quality.json").read_text(encoding="utf-8"))
    if not bool(quality.get("passed")):
        raise RuntimeError("C canonical clean-data gate failed.")
    raw, pool = fg.load_clean_raw()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    base = pd.read_csv(CANDIDATES, dtype={"symbol": str}, encoding="utf-8-sig")
    base = attach_features(base, feature_panel(raw))
    requested = set(sys.argv[1:]) or {"entry_gate", "candidate_rank", "strong_exit"}
    results = []
    if "entry_gate" in requested:
        gate_variants = {rule.key: apply_gate(base, rule) for rule in GATE_RULES}
        results.append(run_module("entry_gate", base, gate_variants, GATE_RULES, raw, pool))
    if "candidate_rank" in requested:
        rank_variants = {rule.key: apply_rank(base, rule) for rule in RANK_RULES}
        results.append(run_module("candidate_rank", base, rank_variants, RANK_RULES, raw, pool))
    if "strong_exit" in requested:
        projected = exit_projection(base, raw)
        exit_variants = {rule.key: extend_exits(projected, rule) for rule in EXIT_RULES}
        results.append(run_module("strong_exit", projected, exit_variants, EXIT_RULES, raw, pool))
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_quality": "C canonical 30-ETF clean whitelist passed",
        "execution": "all features known after signal close; T+1 close execution; exact account fees and 100-share lots",
        "tests": results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
