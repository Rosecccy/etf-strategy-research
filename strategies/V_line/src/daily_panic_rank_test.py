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
from daily_panic_depth_test import attach_depth


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "daily_panic_rank_test"
DEV_META_YEARS = (2019, 2020, 2021, 2022, 2023)
HOLDOUT_YEARS = (2024, 2025, 2026)
WINDOWS: tuple[int | None, ...] = (5, None)
DEPTHS = (0.05, 0.07, 0.10, 0.12, 0.15, 0.17, 0.20)
RECOVERY_FEATURES = ("ma20_gap", "ret1", "ret3", "not_overdeep")
RECOVERY_WEIGHTS = (0.25, 0.50, 0.75, 1.00)


@dataclass(frozen=True)
class Variant:
    depth: float
    recovery: str
    weight: float

    @property
    def key(self) -> str:
        return f"depth{int(self.depth * 100)}:{self.recovery}:w{int(self.weight * 100)}"


VARIANTS = tuple(
    Variant(depth, recovery, weight)
    for depth in DEPTHS
    for recovery in RECOVERY_FEATURES
    for weight in RECOVERY_WEIGHTS
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
    return attach_depth(pd.concat([early, rolling], ignore_index=True, sort=False), raw)


def rank_variant(frame: pd.DataFrame, variant: Variant | None) -> pd.DataFrame:
    if variant is None:
        return frame.copy()
    selected = frame[pd.to_numeric(frame["panic_depth"], errors="coerce").ge(variant.depth)].copy()
    if selected.empty:
        return selected
    selected["not_overdeep"] = -pd.to_numeric(selected["panic_depth"], errors="coerce")
    selected["fear_rank"] = selected.groupby("entry_date")["buy_quality"].rank(pct=True, method="average")
    selected["recovery_rank"] = selected.groupby("entry_date")[variant.recovery].rank(pct=True, method="average")
    selected["original_buy_quality"] = selected["buy_quality"]
    selected["buy_quality"] = 100.0 * (
        (1.0 - variant.weight) * selected["fear_rank"]
        + variant.weight * selected["recovery_rank"]
    )
    selected["rank_variant"] = variant.key
    return selected


def score(stat: dict) -> float:
    return (
        math.log(max(float(stat["final_value"]), 1.0) / 10_000.0)
        + 0.75 * float(np.nan_to_num(stat["win_rate"], nan=0.0))
        + 0.80 * float(stat["max_drawdown"])
    )


def neighbor_count(variant: Variant, eligible: set[str]) -> int:
    di = DEPTHS.index(variant.depth)
    wi = RECOVERY_WEIGHTS.index(variant.weight)
    count = 0
    for other in VARIANTS:
        if other.key not in eligible or other.recovery != variant.recovery:
            continue
        if abs(DEPTHS.index(other.depth) - di) + abs(RECOVERY_WEIGHTS.index(other.weight) - wi) <= 1:
            count += 1
    return count


def training_history(frame: pd.DataFrame, year: int, window: int | None) -> pd.DataFrame:
    start = 2016 if window is None else max(2016, year - window)
    cutoff = pd.Timestamp(year=year, month=1, day=1)
    return frame[
        frame["test_year"].between(start, year - 1)
        & frame["status"].eq("closed")
        & pd.to_datetime(frame["exit_date"]).lt(cutoff)
    ].copy()


def choose_variant(
    formal: pd.DataFrame,
    raw: pd.DataFrame,
    pool: pd.DataFrame,
    year: int,
    window: int | None,
) -> tuple[Variant | None, dict]:
    history = training_history(formal, year, window)
    _, baseline = portfolio_metrics(history, raw, pool)
    if int(baseline["closed_trades"]) < 6:
        return None, {"fallback": True, **baseline}
    rows = []
    for variant in VARIANTS:
        _, stat = portfolio_metrics(rank_variant(history, variant), raw, pool)
        retain = int(stat["closed_trades"]) / max(1, int(baseline["closed_trades"]))
        eligible = bool(
            int(stat["closed_trades"]) >= max(5, math.ceil(int(baseline["closed_trades"]) * 0.60))
            and retain >= 0.60
            and float(stat["final_value"]) > float(baseline["final_value"]) * 1.01
            and float(stat["win_rate"]) > float(baseline["win_rate"]) + 0.005
            and float(stat["max_drawdown"]) >= float(baseline["max_drawdown"]) - 0.02
        )
        rows.append({"variant": variant, "key": variant.key, "eligible": eligible, "retain": retain, "score": score(stat), **stat})
    table = pd.DataFrame(rows)
    eligible = set(table.loc[table["eligible"], "key"])
    if not eligible:
        return None, {"fallback": True, **baseline}
    table["neighbors"] = table["variant"].map(lambda item: neighbor_count(item, eligible))
    stable = table[table["eligible"] & table["neighbors"].ge(2)].sort_values(
        ["score", "final_value", "win_rate"], ascending=False
    )
    if stable.empty:
        return None, {"fallback": True, **baseline}
    winner = stable.iloc[0]
    return winner["variant"], winner.drop(labels=["variant"]).to_dict()


def rolling(
    formal: pd.DataFrame,
    raw: pd.DataFrame,
    pool: pd.DataFrame,
    window: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    choices = []
    for year in range(2019, 2027):
        variant, audit = choose_variant(formal, raw, pool, year, window)
        current = formal[formal["test_year"].eq(year)].copy()
        current = rank_variant(current, variant)
        current["selected_rank_variant"] = "BASE_FEAR" if variant is None else variant.key
        parts.append(current)
        choices.append(
            {
                "year": year,
                "window": "all" if window is None else window,
                "train_start": 2016 if window is None else max(2016, year - window),
                "train_end": year - 1,
                "variant": "BASE_FEAR" if variant is None else variant.key,
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
    _, base_dev = evaluate(formal, raw, pool, DEV_META_YEARS)
    base_holdout_log, base_holdout = evaluate(formal, raw, pool, HOLDOUT_YEARS)
    rows = []
    generated = {}
    for window in WINDOWS:
        key = "all" if window is None else str(window)
        candidates, choices = rolling(formal, raw, pool, window)
        _, dev = evaluate(candidates, raw, pool, DEV_META_YEARS)
        holdout_log, holdout = evaluate(candidates, raw, pool, HOLDOUT_YEARS)
        generated[key] = (candidates, choices, holdout_log)
        rows.append(
            {
                "window": key,
                **{f"dev_{k}": v for k, v in dev.items()},
                **{f"holdout_{k}": v for k, v in holdout.items()},
            }
        )
    table = pd.DataFrame(rows)
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
    _, choices, winner_log = generated[chosen]
    winner_row = table[table["window"].eq(chosen)].iloc[0]
    winner_holdout = {
        key.removeprefix("holdout_"): value
        for key, value in winner_row.to_dict().items()
        if key.startswith("holdout_")
    }
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
        "test": "D2 annual past-only panic-depth plus cross-sectional recovery ranking",
        "execution_unchanged": "signal after close, T+1 close, 90-day/greed70 exit, one full position, exact fees",
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
