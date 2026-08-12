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
OUT = ROOT / "out" / "daily_panic_depth_test"
DEV_META_YEARS = (2019, 2020, 2021, 2022, 2023)
HOLDOUT_YEARS = (2024, 2025, 2026)
WINDOWS: tuple[int | None, ...] = (5, None)


@dataclass(frozen=True)
class Gate:
    feature: str
    direction: str
    threshold: float

    @property
    def key(self) -> str:
        return f"{self.feature}:{self.direction}:{self.threshold:.4f}"


GATES = tuple(
    [Gate("drawdown20", "le", value) for value in (0.0, -0.01, -0.02, -0.03, -0.04, -0.05, -0.07, -0.10, -0.15)]
    + [Gate("ret5", "le", value) for value in (0.02, 0.0, -0.01, -0.02, -0.03, -0.05, -0.07, -0.10)]
    + [Gate("ma20_gap", "le", value) for value in (0.02, 0.01, 0.0, -0.01, -0.02, -0.03, -0.05, -0.08)]
    + [Gate("panic_depth", "ge", value) for value in (0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20)]
)


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def attach_depth(candidates: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in raw.groupby("symbol", sort=False):
        data = group.sort_values("date").copy()
        close = data["close"]
        data["ret1"] = close.pct_change()
        data["ret3"] = close.pct_change(3)
        data["ret5"] = close.pct_change(5)
        data["drawdown20"] = close / close.rolling(20, min_periods=20).max() - 1.0
        data["ma20_gap"] = close / close.rolling(20, min_periods=20).mean() - 1.0
        data["panic_depth"] = np.maximum(-data["drawdown20"], 0.0) + 0.5 * np.maximum(-data["ret5"], 0.0)
        pieces.append(data[["date", "symbol", "ret1", "ret3", "ret5", "drawdown20", "ma20_gap", "panic_depth"]])
    state = pd.concat(pieces, ignore_index=True).rename(columns={"date": "signal_date"})
    return candidates.merge(state, on=["signal_date", "symbol"], how="left")


def apply_gate(frame: pd.DataFrame, gate: Gate | None) -> pd.DataFrame:
    if gate is None:
        return frame.copy()
    values = pd.to_numeric(frame[gate.feature], errors="coerce")
    passed = values.le(gate.threshold) if gate.direction == "le" else values.ge(gate.threshold)
    return frame[passed.fillna(False)].copy()


def metric_score(stat: dict) -> float:
    return (
        math.log(max(float(stat["final_value"]), 1.0) / 10_000.0)
        + 0.75 * float(np.nan_to_num(stat["win_rate"], nan=0.0))
        + 0.80 * float(stat["max_drawdown"])
    )


def adjacent_count(gate: Gate, eligible: set[str]) -> int:
    family = [item for item in GATES if item.feature == gate.feature and item.direction == gate.direction]
    index = family.index(gate)
    count = 0
    for other in family:
        if other.key in eligible and abs(family.index(other) - index) <= 1:
            count += 1
    return count


def choose_gate(
    candidates: pd.DataFrame,
    raw: pd.DataFrame,
    pool: pd.DataFrame,
    year: int,
    window: int | None,
) -> tuple[Gate | None, dict]:
    start = 2016 if window is None else max(2016, year - window)
    cutoff = pd.Timestamp(year=year, month=1, day=1)
    history = candidates[
        candidates["test_year"].between(start, year - 1)
        & candidates["status"].eq("closed")
        & pd.to_datetime(candidates["exit_date"]).lt(cutoff)
    ].copy()
    _, baseline = portfolio_metrics(history, raw, pool)
    if int(baseline["closed_trades"]) < 6:
        return None, {"fallback": True, **baseline}
    rows = []
    for gate in GATES:
        _, stat = portfolio_metrics(apply_gate(history, gate), raw, pool)
        retain = int(stat["closed_trades"]) / max(1, int(baseline["closed_trades"]))
        eligible = bool(
            int(stat["closed_trades"]) >= max(5, math.ceil(int(baseline["closed_trades"]) * 0.60))
            and retain >= 0.60
            and float(stat["final_value"]) > float(baseline["final_value"]) * 1.01
            and float(stat["win_rate"]) > float(baseline["win_rate"]) + 0.005
            and float(stat["max_drawdown"]) >= float(baseline["max_drawdown"]) - 0.02
        )
        rows.append({"gate": gate, "key": gate.key, "eligible": eligible, "retain": retain, "score": metric_score(stat), **stat})
    table = pd.DataFrame(rows)
    eligible_keys = set(table.loc[table["eligible"], "key"])
    if not eligible_keys:
        return None, {"fallback": True, **baseline}
    table["neighbors"] = table["gate"].map(lambda gate: adjacent_count(gate, eligible_keys))
    stable = table[table["eligible"] & table["neighbors"].ge(2)].sort_values(
        ["score", "final_value", "win_rate"], ascending=False
    )
    if stable.empty:
        return None, {"fallback": True, **baseline}
    winner = stable.iloc[0]
    return winner["gate"], winner.drop(labels=["gate"]).to_dict()


def rolling_candidates(
    candidates: pd.DataFrame,
    raw: pd.DataFrame,
    pool: pd.DataFrame,
    window: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    choices = []
    for year in range(2019, 2027):
        gate, audit = choose_gate(candidates, raw, pool, year, window)
        current = candidates[candidates["test_year"].eq(year)].copy()
        current = apply_gate(current, gate)
        current["depth_gate"] = "KEEP_ALL" if gate is None else gate.key
        parts.append(current)
        choices.append(
            {
                "year": year,
                "window": "all" if window is None else window,
                "train_start": 2016 if window is None else max(2016, year - window),
                "train_end": year - 1,
                "gate": "KEEP_ALL" if gate is None else gate.key,
                "train_score": audit.get("score"),
                "train_trades": audit.get("closed_trades"),
                "neighbors": audit.get("neighbors", 0),
            }
        )
    return pd.concat(parts, ignore_index=True), pd.DataFrame(choices)


def evaluate(candidates: pd.DataFrame, raw: pd.DataFrame, pool: pd.DataFrame, years: tuple[int, ...]) -> tuple[pd.DataFrame, dict]:
    return portfolio_metrics(candidates[candidates["test_year"].isin(years)].copy(), raw, pool)


def main() -> None:
    configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    adjusted, raw, pool = base.prepare()
    formal, base_choices = base.rolling_candidates(adjusted, raw, pool, None)
    formal = attach_depth(formal, raw)
    _, base_dev = evaluate(formal, raw, pool, DEV_META_YEARS)
    base_holdout_log, base_holdout = evaluate(formal, raw, pool, HOLDOUT_YEARS)
    variants = []
    generated = {}
    for window in WINDOWS:
        key = "all" if window is None else str(window)
        candidates, choices = rolling_candidates(formal, raw, pool, window)
        _, dev = evaluate(candidates, raw, pool, DEV_META_YEARS)
        holdout_log, holdout = evaluate(candidates, raw, pool, HOLDOUT_YEARS)
        generated[key] = (candidates, choices, holdout_log)
        variants.append(
            {
                "window": key,
                **{f"dev_{k}": v for k, v in dev.items()},
                **{f"holdout_{k}": v for k, v in holdout.items()},
            }
        )
    table = pd.DataFrame(variants)
    dev_ok = table[
        table["dev_final_value"].gt(float(base_dev["final_value"]))
        & table["dev_win_rate"].gt(float(base_dev["win_rate"]))
    ].copy()
    if dev_ok.empty:
        chosen = "all"
    else:
        dev_ok["score"] = (
            np.log(dev_ok["dev_final_value"] / 10_000.0)
            + 0.75 * dev_ok["dev_win_rate"]
            + 0.80 * dev_ok["dev_max_drawdown"]
        )
        chosen = str(dev_ok.sort_values("score", ascending=False).iloc[0]["window"])
    candidates, choices, winner_log = generated[chosen]
    _, winner_holdout = evaluate(candidates, raw, pool, HOLDOUT_YEARS)
    robust = int(
        (
            table["holdout_final_value"].gt(float(base_holdout["final_value"]))
            & table["holdout_win_rate"].gt(float(base_holdout["win_rate"]))
            & table["holdout_max_drawdown"].ge(float(base_holdout["max_drawdown"]))
        ).sum()
    )
    promoted = bool(
        float(winner_holdout["final_value"]) > float(base_holdout["final_value"])
        and float(winner_holdout["win_rate"]) > float(base_holdout["win_rate"])
        and float(winner_holdout["max_drawdown"]) >= float(base_holdout["max_drawdown"])
        and int(winner_holdout["closed_trades"]) >= 6
        and robust >= 2
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test": "D2 annual past-only panic-depth gate",
        "control_variable": "entry panic depth only; thresholds, exits, portfolio and fees fixed",
        "development_selected_window": chosen,
        "baseline_development": base_dev,
        "baseline_holdout": base_holdout,
        "winner_holdout": winner_holdout,
        "holdout_neighbor_windows_both_improved": robust,
        "promoted": promoted,
    }
    table.to_csv(OUT / "variants.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(OUT / "winner_choices.csv", index=False, encoding="utf-8-sig")
    winner_log.to_csv(OUT / "winner_holdout_trades.csv", index=False, encoding="utf-8-sig")
    base_holdout_log.to_csv(OUT / "baseline_holdout_trades.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
