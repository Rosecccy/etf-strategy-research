from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "hierarchical_trade_quality_test"
C_TRADES = ROOT / "fit" / "hybrid_formal" / "final_trades.csv"
S_TRADES = ROOT.parent / "S" / "fit" / "selector" / "final_trades.csv"
DEV_META_YEARS = {2019, 2020, 2021, 2022, 2023}
HOLDOUT_YEARS = {2024, 2025, 2026}
WINDOWS: tuple[int | None, ...] = (3, 5, None)


@dataclass(frozen=True)
class Gate:
    scope: str
    group: str
    lookback: int | None
    min_history: int
    win_min: float
    mean_min: float

    @property
    def key(self) -> str:
        return (
            f"{self.scope}:{self.group}:lb{self.lookback or 'all'}:n{self.min_history}:"
            f"w{int(self.win_min * 100)}:m{int(self.mean_min * 1000)}"
        )


GATES = tuple(
    Gate(scope, group, lookback, minimum, win_min, mean_min)
    for scope in ("all", "main")
    for group in ("category", "symbol")
    for lookback in (None, 5, 10)
    for minimum in (2, 3, 5)
    for win_min in (0.40, 0.50, 0.60, 0.70)
    for mean_min in (-0.01, 0.0, 0.01, 0.02)
)


def read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    for column in ("entry_date", "exit_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    frame["original_year"] = frame["entry_date"].dt.year
    return frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def apply_gate(frame: pd.DataFrame, gate: Gate) -> pd.DataFrame:
    rows = []
    ordered = frame.sort_values(["entry_date", "symbol"])
    for _, trade in ordered.iterrows():
        if gate.scope == "main" and str(trade["source"]) != "主策略":
            rows.append(trade.to_dict())
            continue
        history = ordered[
            ordered["exit_date"].lt(trade["entry_date"])
            & ordered[gate.group].astype(str).eq(str(trade[gate.group]))
        ].sort_values("exit_date")
        if gate.lookback is not None:
            history = history.tail(gate.lookback)
        returns = pd.to_numeric(history["ret"], errors="coerce").dropna()
        count = int(len(returns))
        if count < gate.min_history:
            accepted = True
            win = np.nan
            mean = np.nan
        else:
            win = float((returns > 0).mean())
            mean = float(returns.mean())
            accepted = bool(win >= gate.win_min and mean >= gate.mean_min)
        if accepted:
            item = trade.to_dict()
            item.update(
                {
                    "quality_gate": gate.key,
                    "quality_history": count,
                    "quality_win": win,
                    "quality_mean": mean,
                }
            )
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


def neighbor_count(gate: Gate, eligible: set[str]) -> int:
    count = 0
    for other in GATES:
        if other.key not in eligible:
            continue
        if (
            other.scope != gate.scope
            or other.group != gate.group
            or other.lookback != gate.lookback
            or other.min_history != gate.min_history
        ):
            continue
        wi = (0.40, 0.50, 0.60, 0.70).index(gate.win_min)
        wj = (0.40, 0.50, 0.60, 0.70).index(other.win_min)
        mi = (-0.01, 0.0, 0.01, 0.02).index(gate.mean_min)
        mj = (-0.01, 0.0, 0.01, 0.02).index(other.mean_min)
        if abs(wi - wj) + abs(mi - mj) <= 1:
            count += 1
    return count


def choose_gate(
    baseline: pd.DataFrame,
    logs: dict[str, pd.DataFrame],
    year: int,
    window: int | None,
) -> tuple[Gate | None, dict]:
    min_year = int(baseline["original_year"].min())
    start = min_year if window is None else max(min_year, year - window)
    years = set(range(start, year))
    base = metrics(baseline, years)
    if base["trades"] < 15:
        return None, {"fallback": True, **base}
    rows = []
    for gate in GATES:
        stat = metrics(logs[gate.key], years)
        retain = stat["trades"] / max(1, base["trades"])
        eligible = bool(
            stat["trades"] >= 15
            and retain >= 0.75
            and stat["final_1000"] > base["final_1000"] * 1.01
            and stat["win_rate"] > base["win_rate"] + 0.005
            and stat["max_drawdown"] >= base["max_drawdown"] - 0.01
        )
        rows.append({"gate": gate, "key": gate.key, "eligible": eligible, "retain": retain, "score": score(stat), **stat})
    table = pd.DataFrame(rows)
    eligible = set(table.loc[table["eligible"], "key"])
    if not eligible:
        return None, {"fallback": True, **base}
    table["neighbors"] = table["gate"].map(lambda item: neighbor_count(item, eligible))
    stable = table[table["eligible"] & table["neighbors"].ge(2)].sort_values(
        ["score", "final_1000", "win_rate"], ascending=False
    )
    if stable.empty:
        return None, {"fallback": True, **base}
    winner = stable.iloc[0]
    return winner["gate"], winner.drop(labels=["gate"]).to_dict()


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
        gate, audit = choose_gate(baseline, logs, year, window)
        source = baseline if gate is None else logs[gate.key]
        parts.append(source[source["original_year"].eq(year)].copy())
        choices.append(
            {
                "year": year,
                "window": "all" if window is None else window,
                "train_start": min_year if window is None else max(min_year, year - window),
                "train_end": year - 1,
                "gate": "KEEP_ALL" if gate is None else gate.key,
                "train_score": audit.get("score"),
                "train_trades": audit.get("trades"),
                "neighbors": audit.get("neighbors", 0),
            }
        )
    return pd.concat(parts, ignore_index=True, sort=False).sort_values(["entry_date", "symbol"]), pd.DataFrame(choices)


def run_line(name: str, baseline: pd.DataFrame) -> dict:
    logs = {gate.key: apply_gate(baseline, gate) for gate in GATES}
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
    c = read(C_TRADES)
    s = read(S_TRADES)
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test": "C/S annual past-only hierarchical category/symbol quality gate",
        "shadow_learning": "rejected virtual trades remain available as completed historical labels",
        "C": run_line("C", c),
        "S": run_line("S", s),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
