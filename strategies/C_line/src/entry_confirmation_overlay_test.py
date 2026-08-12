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


OUT = ROOT / "fit" / "entry_confirmation_overlay_test"
C_TRADES = ROOT / "fit" / "hybrid_formal" / "final_trades.csv"
S_TRADES = PROJECT / "S" / "fit" / "selector" / "final_trades.csv"
HOLDOUT_YEARS = {2024, 2025, 2026}
DEV_META_YEARS = {2019, 2020, 2021, 2022, 2023}
WINDOWS: tuple[int | None, ...] = (3, 5, None)
WAITS = (1, 2, 3, 5)
CONDITIONS = {
    "ret1": (-0.01, 0.0, 0.005, 0.01, 0.02),
    "ret3": (-0.02, -0.01, 0.0, 0.01, 0.02),
    "rebound3": (0.0, 0.005, 0.01, 0.02),
    "ma5_gap": (-0.02, -0.01, 0.0, 0.01),
    "ma10_gap": (-0.02, -0.01, 0.0, 0.01),
    "signal_recovery": (-0.01, 0.0, 0.005, 0.01, 0.02),
}


@dataclass(frozen=True)
class Rule:
    scope: str
    condition: str
    threshold: float
    wait: int

    @property
    def key(self) -> str:
        value = int(round(self.threshold * 1000))
        return f"{self.scope}:{self.condition}:t{value}:w{self.wait}"


def read_trades(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    for column in ("entry_date", "exit_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in ("ret", "entry_close", "exit_close"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame["original_year"] = frame["entry_date"].dt.year
    frame["original_entry_date"] = frame["entry_date"]
    frame["original_entry_close"] = frame["entry_close"]
    return frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def build_panel() -> pd.DataFrame:
    raw, _ = fg.load_clean_raw()
    pieces: list[pd.DataFrame] = []
    for _, group in raw.groupby("symbol", sort=False):
        data = group.sort_values("date").copy()
        close = data["close"]
        data["ret1"] = close.pct_change()
        data["ret3"] = close.pct_change(3)
        data["rebound3"] = close / close.rolling(3, min_periods=3).min() - 1.0
        data["ma5_gap"] = close / close.rolling(5, min_periods=5).mean() - 1.0
        data["ma10_gap"] = close / close.rolling(10, min_periods=10).mean() - 1.0
        pieces.append(data)
    return pd.concat(pieces, ignore_index=True).sort_values(["symbol", "date"])


def is_target(source: object, scope: str) -> bool:
    main = str(source) == "主策略"
    return scope == "all" or (scope == "main" and main) or (scope == "fallback" and not main)


def transform_trade(trade: pd.Series, history: pd.DataFrame, rule: Rule) -> dict | None:
    item = trade.to_dict()
    if not is_target(trade.get("source"), rule.scope):
        item["confirmation_rule"] = "not_targeted"
        return item
    before = history[history["date"].lt(trade["original_entry_date"])]
    if before.empty:
        return None
    decision = before.iloc[-1]
    window = history[history["date"].ge(decision["date"])].head(rule.wait + 1).copy()
    if window.empty:
        return None
    window["signal_recovery"] = window["close"] / float(decision["close"]) - 1.0
    passed = window[pd.to_numeric(window[rule.condition], errors="coerce").ge(rule.threshold)]
    if passed.empty:
        return None
    confirmation = passed.iloc[0]
    later = history[history["date"].gt(confirmation["date"])]
    if later.empty:
        return None
    entry = later.iloc[0]
    if pd.Timestamp(entry["date"]) >= pd.Timestamp(trade["exit_date"]):
        return None
    original_entry = float(trade["original_entry_close"])
    new_entry = float(entry["close"])
    original_ret = float(trade["ret"])
    item["entry_date"] = pd.Timestamp(entry["date"])
    item["entry_close"] = new_entry
    item["ret"] = (1.0 + original_ret) * original_entry / new_entry - 1.0
    item["confirmation_date"] = pd.Timestamp(confirmation["date"])
    item["confirmation_value"] = float(confirmation[rule.condition])
    item["confirmation_rule"] = rule.key
    return item


def enforce_single_position(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    rows: list[pd.Series] = []
    busy = pd.Timestamp.min
    for _, trade in frame.sort_values(["entry_date", "symbol"]).iterrows():
        if pd.Timestamp(trade["entry_date"]) < busy:
            continue
        rows.append(trade)
        busy = pd.Timestamp(trade["exit_date"])
    return pd.DataFrame(rows).reset_index(drop=True)


def build_logs(frame: pd.DataFrame, panel: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Rule]]:
    histories = {symbol: group.sort_values("date") for symbol, group in panel.groupby("symbol")}
    logs = {"KEEP_ALL": frame.copy()}
    rules: dict[str, Rule] = {}
    for scope in ("all", "main", "fallback"):
        for condition, thresholds in CONDITIONS.items():
            for threshold in thresholds:
                for wait in WAITS:
                    rule = Rule(scope, condition, threshold, wait)
                    rows = []
                    for _, trade in frame.iterrows():
                        history = histories.get(str(trade["symbol"]))
                        if history is None:
                            continue
                        result = transform_trade(trade, history, rule)
                        if result is not None:
                            rows.append(result)
                    if not rows:
                        continue
                    transformed = enforce_single_position(pd.DataFrame(rows))
                    logs[rule.key] = transformed
                    rules[rule.key] = rule
    return logs, rules


def metrics(frame: pd.DataFrame, years: set[int] | None = None) -> dict[str, float | int]:
    local = frame if years is None else frame[frame["original_year"].isin(years)]
    local = local.sort_values(["entry_date", "symbol"])
    returns = pd.to_numeric(local["ret"], errors="coerce").dropna()
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


def neighbor_count(key: str, eligible: set[str], rules: dict[str, Rule]) -> int:
    rule = rules[key]
    count = 0
    for other_key in eligible:
        other = rules[other_key]
        if other.scope != rule.scope or other.condition != rule.condition:
            continue
        thresholds = CONDITIONS[rule.condition]
        ti = thresholds.index(rule.threshold)
        oi = thresholds.index(other.threshold)
        wi = WAITS.index(rule.wait)
        wj = WAITS.index(other.wait)
        if abs(ti - oi) + abs(wi - wj) <= 1:
            count += 1
    return count


def select_rule(
    baseline: pd.DataFrame,
    logs: dict[str, pd.DataFrame],
    rules: dict[str, Rule],
    train_years: set[int],
) -> tuple[str, dict]:
    base = metrics(baseline, train_years)
    rows = []
    for key, log in logs.items():
        if key == "KEEP_ALL":
            continue
        stat = metrics(log, train_years)
        retain = stat["trades"] / max(base["trades"], 1)
        eligible = (
            stat["trades"] >= 12
            and retain >= 0.72
            and stat["final_1000"] > base["final_1000"] * 1.01
            and stat["win_rate"] > base["win_rate"] + 0.005
        )
        score = (
            math.log(max(stat["final_1000"], 1.0) / 1000.0)
            + 0.65 * stat["win_rate"]
            + 0.80 * stat["max_drawdown"]
        )
        rows.append({"key": key, "eligible": eligible, "retain": retain, "score": score, **stat})
    table = pd.DataFrame(rows)
    eligible_keys = set(table.loc[table["eligible"], "key"])
    if not eligible_keys:
        return "KEEP_ALL", {"fallback": True, **base}
    table["neighbors"] = table["key"].map(
        lambda key: neighbor_count(key, eligible_keys, rules) if key in eligible_keys else 0
    )
    candidates = table[table["eligible"] & table["neighbors"].ge(2)].sort_values(
        ["score", "final_1000", "win_rate"], ascending=False
    )
    if candidates.empty:
        return "KEEP_ALL", {"fallback": True, **base}
    winner = candidates.iloc[0]
    return str(winner["key"]), winner.to_dict()


def rolling(
    baseline: pd.DataFrame,
    logs: dict[str, pd.DataFrame],
    rules: dict[str, Rule],
    window: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    min_year = int(baseline["original_year"].min())
    first_year = max(2019, min_year + 3)
    parts = [baseline[baseline["original_year"].lt(first_year)].copy()]
    choices = []
    for year in range(first_year, 2027):
        start = min_year if window is None else max(min_year, year - window)
        years = set(range(start, year))
        key, audit = select_rule(baseline, logs, rules, years)
        current = logs[key]
        parts.append(current[current["original_year"].eq(year)].copy())
        choices.append(
            {
                "year": year,
                "window": "all" if window is None else window,
                "train_start": start,
                "train_end": year - 1,
                "rule": key,
                "train_score": audit.get("score"),
                "train_trades": audit.get("trades"),
                "neighbors": audit.get("neighbors", 0),
            }
        )
    return enforce_single_position(pd.concat(parts, ignore_index=True, sort=False)), pd.DataFrame(choices)


def run_line(name: str, baseline: pd.DataFrame, panel: pd.DataFrame) -> dict:
    logs, rules = build_logs(baseline, panel)
    variants = []
    generated = {}
    base_full = metrics(baseline)
    base_dev = metrics(baseline, DEV_META_YEARS)
    base_holdout = metrics(baseline, HOLDOUT_YEARS)
    for window in WINDOWS:
        key = "all" if window is None else str(window)
        trades, choices = rolling(baseline, logs, rules, window)
        generated[key] = (trades, choices)
        variants.append(
            {
                "window": key,
                **{f"full_{k}": v for k, v in metrics(trades).items()},
                **{f"dev_{k}": v for k, v in metrics(trades, DEV_META_YEARS).items()},
                **{f"holdout_{k}": v for k, v in metrics(trades, HOLDOUT_YEARS).items()},
            }
        )
    table = pd.DataFrame(variants)
    development = table[
        table["dev_final_1000"].gt(base_dev["final_1000"])
        & table["dev_win_rate"].gt(base_dev["win_rate"])
    ].copy()
    if development.empty:
        chosen = "all"
    else:
        development["score"] = (
            np.log(development["dev_final_1000"] / 1000.0)
            + 0.65 * development["dev_win_rate"]
            + 0.80 * development["dev_max_drawdown"]
        )
        chosen = str(development.sort_values("score", ascending=False).iloc[0]["window"])
    winner, choices = generated[chosen]
    win_full = metrics(winner)
    win_holdout = metrics(winner, HOLDOUT_YEARS)
    robust = int(
        (
            table["holdout_final_1000"].gt(base_holdout["final_1000"])
            & table["holdout_win_rate"].gt(base_holdout["win_rate"])
        ).sum()
    )
    promoted = bool(
        win_full["final_1000"] > base_full["final_1000"]
        and win_full["win_rate"] > base_full["win_rate"]
        and win_holdout["final_1000"] > base_holdout["final_1000"]
        and win_holdout["win_rate"] > base_holdout["win_rate"]
        and win_holdout["max_drawdown"] >= base_holdout["max_drawdown"] - 0.01
        and robust >= 2
    )
    table.to_csv(OUT / f"{name.lower()}_variants.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(OUT / f"{name.lower()}_choices.csv", index=False, encoding="utf-8-sig")
    winner.to_csv(OUT / f"{name.lower()}_winner_trades.csv", index=False, encoding="utf-8-sig")
    return {
        "line": name,
        "candidate_rule_count": len(rules),
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
    panel = build_panel()
    c = read_trades(C_TRADES)
    s = read_trades(S_TRADES)
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test": "annual past-only stop-and-rebound entry confirmation overlay",
        "fixed": "original signals, exits, single-position discipline and return caliber",
        "execution": "confirmation after close, buy at next trading-day close",
        "C": run_line("C", c, panel),
        "S": run_line("S", s, panel),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
