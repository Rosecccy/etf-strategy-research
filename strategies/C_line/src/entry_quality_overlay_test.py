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
import fear_greed_weight_sweep as sweep  # noqa: E402


OUT = ROOT / "fit" / "entry_quality_overlay_test"
C_TRADES = ROOT / "fit" / "hybrid_formal" / "final_trades.csv"
S_TRADES = PROJECT / "S" / "fit" / "selector" / "final_trades.csv"
HOLDOUT_YEARS = {2024, 2025, 2026}
DEV_META_YEARS = {2019, 2020, 2021, 2022, 2023}
WINDOWS: tuple[int | None, ...] = (3, 5, None)
QUANTILES = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
FEATURES = (
    "ret1",
    "ret3",
    "ret5",
    "ret10",
    "ret20",
    "ret60",
    "ma5_gap",
    "ma10_gap",
    "ma20_gap",
    "ma60_gap",
    "drawdown20",
    "drawdown60",
    "rebound5",
    "vol20",
    "volume_ratio20",
    "symbol_fear",
    "market_fear",
)


@dataclass(frozen=True)
class Gate:
    scope: str
    feature: str
    direction: str
    quantile: float

    @property
    def key(self) -> str:
        return f"{self.scope}:{self.feature}:{self.direction}:q{int(self.quantile * 100)}"


def read_trades(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    frame["exit_date"] = pd.to_datetime(frame["exit_date"], errors="coerce")
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    frame["year"] = frame["entry_date"].dt.year
    return frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def build_state_panel() -> pd.DataFrame:
    raw, _ = fg.load_clean_raw()
    raw = raw.sort_values(["symbol", "date"]).copy()
    symbol_fear = fg.build_symbol_fear(raw)
    sentiment = fg.build_sentiment(raw)
    sentiment["market_fear"] = sweep.composite(
        sentiment["fear_core"], sentiment["fear_qvix"], 0.2
    )
    pieces: list[pd.DataFrame] = []
    for _, group in raw.groupby("symbol", sort=False):
        data = group.sort_values("date").copy()
        close = data["close"]
        returns = close.pct_change()
        for days in (1, 3, 5, 10, 20, 60):
            data[f"ret{days}"] = close.pct_change(days)
        for days in (5, 10, 20, 60):
            data[f"ma{days}_gap"] = close / close.rolling(days, min_periods=days).mean() - 1.0
        for days in (20, 60):
            data[f"drawdown{days}"] = close / close.rolling(days, min_periods=days).max() - 1.0
        data["rebound5"] = close / close.rolling(5, min_periods=5).min() - 1.0
        data["vol20"] = returns.rolling(20, min_periods=20).std() * np.sqrt(252)
        data["volume_ratio20"] = data["volume"] / data["volume"].rolling(20, min_periods=20).median()
        pieces.append(data)
    panel = pd.concat(pieces, ignore_index=True)
    panel = panel.merge(symbol_fear, on=["date", "symbol"], how="left")
    panel = panel.merge(sentiment[["date", "market_fear"]], on="date", how="left")
    return panel.sort_values(["symbol", "date"]).reset_index(drop=True)


def attach_pre_entry_state(trades: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = {symbol: group.sort_values("date") for symbol, group in panel.groupby("symbol")}
    for _, trade in trades.iterrows():
        item = trade.to_dict()
        history = grouped.get(str(trade["symbol"]))
        if history is None:
            rows.append(item)
            continue
        prior = history[history["date"].lt(trade["entry_date"])]
        if prior.empty:
            rows.append(item)
            continue
        state = prior.iloc[-1]
        item["state_date"] = state["date"]
        for feature in FEATURES:
            item[feature] = state.get(feature, np.nan)
        rows.append(item)
    return pd.DataFrame(rows).sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def metrics(frame: pd.DataFrame, years: set[int] | None = None) -> dict[str, float | int]:
    local = frame if years is None else frame[frame["year"].isin(years)]
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


def targeted(frame: pd.DataFrame, scope: str) -> pd.Series:
    is_main = frame["source"].astype(str).eq("主策略")
    if scope == "main":
        return is_main
    if scope == "fallback":
        return ~is_main
    return pd.Series(True, index=frame.index)


def apply_gate(frame: pd.DataFrame, gate: Gate, threshold: float) -> pd.DataFrame:
    target = targeted(frame, gate.scope)
    values = pd.to_numeric(frame[gate.feature], errors="coerce")
    passed = values.ge(threshold) if gate.direction == "ge" else values.le(threshold)
    keep = (~target) | (target & passed.fillna(False))
    return frame[keep].copy()


def candidate_table(train: pd.DataFrame) -> pd.DataFrame:
    baseline = metrics(train)
    rows: list[dict] = []
    for scope in ("all", "main", "fallback"):
        for feature in FEATURES:
            values = pd.to_numeric(train.loc[targeted(train, scope), feature], errors="coerce").dropna()
            if len(values) < 10 or values.nunique() < 5:
                continue
            for direction in ("ge", "le"):
                for quantile in QUANTILES:
                    gate = Gate(scope, feature, direction, quantile)
                    threshold = float(values.quantile(quantile))
                    kept = apply_gate(train, gate, threshold)
                    stat = metrics(kept)
                    retain = stat["trades"] / max(1, baseline["trades"])
                    improves = (
                        retain >= 0.72
                        and stat["trades"] >= 12
                        and stat["final_1000"] > baseline["final_1000"] * 1.01
                        and stat["win_rate"] > baseline["win_rate"] + 0.005
                    )
                    score = (
                        math.log(max(stat["final_1000"], 1.0) / 1000.0)
                        + 0.65 * stat["win_rate"]
                        + 0.80 * stat["max_drawdown"]
                    )
                    rows.append(
                        {
                            "gate": gate,
                            "gate_key": gate.key,
                            "threshold": threshold,
                            "retain": retain,
                            "eligible": improves,
                            "score": score,
                            **stat,
                        }
                    )
    return pd.DataFrame(rows)


def select_gate(train: pd.DataFrame) -> tuple[Gate | None, float | None, dict]:
    table = candidate_table(train)
    eligible = table[table["eligible"]].copy()
    if eligible.empty:
        return None, None, {"fallback": True, **metrics(train)}
    eligible = eligible.sort_values(["score", "final_1000", "win_rate"], ascending=False)
    for _, row in eligible.iterrows():
        gate: Gate = row["gate"]
        adjacent = [q for q in QUANTILES if abs(q - gate.quantile) <= 0.100001]
        neighbor = table[
            table["eligible"]
            & table["gate_key"].str.startswith(f"{gate.scope}:{gate.feature}:{gate.direction}:")
            & table["gate"].map(lambda value: value.quantile in adjacent)
        ]
        if len(neighbor) >= 2:
            audit = row.drop(labels=["gate"]).to_dict()
            audit["neighbor_count"] = int(len(neighbor))
            return gate, float(row["threshold"]), audit
    return None, None, {"fallback": True, **metrics(train)}


def train_slice(frame: pd.DataFrame, year: int, window: int | None) -> pd.DataFrame:
    start = int(frame["year"].min()) if window is None else max(int(frame["year"].min()), year - window)
    return frame[frame["year"].between(start, year - 1)].copy()


def rolling_overlay(frame: pd.DataFrame, window: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_year = max(2019, int(frame["year"].min()) + 3)
    untouched = frame[frame["year"].lt(first_year)].copy()
    parts = [untouched]
    choices: list[dict] = []
    for year in range(first_year, 2027):
        train = train_slice(frame, year, window)
        gate, threshold, audit = select_gate(train)
        current = frame[frame["year"].eq(year)].copy()
        if gate is not None and threshold is not None:
            current = apply_gate(current, gate, threshold)
        parts.append(current)
        choices.append(
            {
                "year": year,
                "window": "all" if window is None else window,
                "train_start": int(train["year"].min()) if not train.empty else None,
                "train_end": year - 1,
                "gate": "KEEP_ALL" if gate is None else gate.key,
                "threshold": threshold,
                "train_score": audit.get("score"),
                "train_trades": audit.get("trades"),
                "neighbor_count": audit.get("neighbor_count", 0),
            }
        )
    result = pd.concat(parts, ignore_index=True, sort=False).sort_values(["entry_date", "symbol"])
    return result, pd.DataFrame(choices)


def run_line(name: str, frame: pd.DataFrame) -> tuple[dict, dict[str, pd.DataFrame]]:
    baseline_full = metrics(frame)
    baseline_dev = metrics(frame, DEV_META_YEARS)
    baseline_holdout = metrics(frame, HOLDOUT_YEARS)
    variants: list[dict] = []
    generated: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for window in WINDOWS:
        key = "all" if window is None else str(window)
        trades, choices = rolling_overlay(frame, window)
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
    table["dev_both_improved"] = (
        table["dev_final_1000"].gt(baseline_dev["final_1000"])
        & table["dev_win_rate"].gt(baseline_dev["win_rate"])
    )
    eligible = table[table["dev_both_improved"]].copy()
    if eligible.empty:
        chosen = "all"
    else:
        eligible["dev_score"] = (
            np.log(eligible["dev_final_1000"] / 1000.0)
            + 0.65 * eligible["dev_win_rate"]
            + 0.80 * eligible["dev_max_drawdown"]
        )
        chosen = str(eligible.sort_values("dev_score", ascending=False).iloc[0]["window"])
    winner, choices = generated[chosen]
    winner_full = metrics(winner)
    winner_holdout = metrics(winner, HOLDOUT_YEARS)
    robust_count = int(
        (
            table["holdout_final_1000"].gt(baseline_holdout["final_1000"])
            & table["holdout_win_rate"].gt(baseline_holdout["win_rate"])
        ).sum()
    )
    promoted = bool(
        winner_full["final_1000"] > baseline_full["final_1000"]
        and winner_full["win_rate"] > baseline_full["win_rate"]
        and winner_holdout["final_1000"] > baseline_holdout["final_1000"]
        and winner_holdout["win_rate"] > baseline_holdout["win_rate"]
        and winner_holdout["max_drawdown"] >= baseline_holdout["max_drawdown"] - 0.01
        and robust_count >= 2
    )
    summary = {
        "line": name,
        "selected_window_on_development": chosen,
        "baseline_full": baseline_full,
        "winner_full": winner_full,
        "baseline_holdout": baseline_holdout,
        "winner_holdout": winner_holdout,
        "holdout_neighbor_windows_both_improved": robust_count,
        "promoted": promoted,
    }
    table.to_csv(OUT / f"{name.lower()}_variants.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(OUT / f"{name.lower()}_choices.csv", index=False, encoding="utf-8-sig")
    winner.to_csv(OUT / f"{name.lower()}_winner_trades.csv", index=False, encoding="utf-8-sig")
    return summary, {key: value[0] for key, value in generated.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = build_state_panel()
    c = attach_pre_entry_state(read_trades(C_TRADES), panel)
    s = attach_pre_entry_state(read_trades(S_TRADES), panel)
    c.to_csv(OUT / "c_state_audit.csv", index=False, encoding="utf-8-sig")
    s.to_csv(OUT / "s_state_audit.csv", index=False, encoding="utf-8-sig")
    c_summary, _ = run_line("C", c)
    s_summary, _ = run_line("S", s)
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test": "annual past-only causal pre-entry quality overlay",
        "execution_unchanged": "existing entry, exit, position and fee rules",
        "development_meta_years": sorted(DEV_META_YEARS),
        "holdout_years": sorted(HOLDOUT_YEARS),
        "hard_gate": "full and holdout profit plus win rate must all improve; at least two window neighbors",
        "C": c_summary,
        "S": s_summary,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
