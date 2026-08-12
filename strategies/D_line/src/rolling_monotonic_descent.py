from __future__ import annotations

"""Strict rolling selection of a score that rises with a continuing selloff.

After a causal low-point signal, a lower close inside the next few sessions
receives a monotonic floor score.  The floor is based only on the current
drawdown from that already-known signal, so it cannot see a future rebound.
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd

import rolling_extrema5_online as online
from fit_dividend_extrema5 import OUT, PROJECT, SYMBOL


BASE_RATE = 0.10
PREDICTION_YEARS = range(2020, 2026)
HISTORY_WINDOWS = (3, 5, 7, 999)


def add_leg_state(frame: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Anchor every new selloff leg at the most recent already-fired base signal."""
    result = frame.sort_values("date").copy().reset_index(drop=True)
    anchor_index = -10_000
    anchor_close = np.nan
    running_low = np.nan
    anchor = np.full(len(result), np.nan)
    new_low = np.zeros(len(result), dtype=bool)
    for index, row in result.iterrows():
        if index - anchor_index <= lookback:
            anchor[index] = anchor_close
            if float(row["close"]) < running_low:
                new_low[index] = True
                running_low = float(row["close"])
        if bool(row["base_signal"]):
            anchor_index = index
            anchor_close = float(row["close"])
            running_low = anchor_close
    result["leg_anchor_close"] = anchor
    result["leg_drop"] = result["close"] / result["leg_anchor_close"] - 1.0
    result["leg_new_low"] = new_low
    return result


def adjusted(state: pd.DataFrame, start: float, span: float, floor: float, ceiling: float) -> pd.DataFrame:
    """Create ranks with an explicitly monotonic continuing-selloff component."""
    rows = []
    for year in sorted(state["year"].unique()):
        calibration = state.loc[state["year"] == year - 1]
        test = state.loc[state["year"] == year]
        if test.empty:
            continue
        def raw_score(part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            depth = ((-part["leg_drop"].to_numpy()) - start) / span
            depth = np.clip(np.nan_to_num(depth, nan=-1.0), 0.0, 1.0)
            leg_score = floor + (ceiling - floor) * depth
            eligible = np.isfinite(part["leg_drop"].to_numpy()) & (depth > 0) & part["leg_new_low"].to_numpy()
            score = np.maximum(part["rank"].to_numpy(), np.where(eligible, leg_score, -np.inf))
            return score, depth
        test_score, depth = raw_score(test)
        if calibration.empty:
            # The first scored year has no prior scored year. Its base rank is
            # already strictly online, so retain it as the initial reference.
            final_rank = np.clip(test_score, 0.0, 1.0)
        else:
            calibration_score, _ = raw_score(calibration)
            final_rank = online.relative_rank(test_score, calibration_score)
        rows.append(pd.DataFrame({
            "date": test["date"].to_numpy(), "year": year, "actual": test["actual"].to_numpy(),
            "base_rank": test["rank"].to_numpy(), "base_signal": test["base_signal"].to_numpy(), "W_RSI6": test["W_RSI6"].to_numpy(),
            "leg_drop": test["leg_drop"].to_numpy(), "leg_depth": depth, "score": test_score,
            "rank": final_rank,
            "start": start, "span": span, "floor": floor, "ceiling": ceiling,
        }))
    return pd.concat(rows, ignore_index=True)


def signal(frame: pd.DataFrame, rate: float) -> pd.Series:
    """Keep every validated base signal; monotonic scoring can only add signals."""
    enhanced = (frame["rank"] >= 1.0 - rate) & (frame["W_RSI6"] <= 45)
    return frame["base_signal"] | enhanced


def metrics(frame: pd.DataFrame, chosen: pd.Series) -> dict[str, float | int]:
    total = int(chosen.sum())
    hits = int(frame.loc[chosen, "actual"].sum())
    positives = int(frame["actual"].sum())
    annual = []
    for _, part in frame.groupby("year"):
        mask = chosen.loc[part.index]
        annual.append(float(part.loc[mask, "actual"].mean()) if mask.any() else 0.0)
    return {
        "signals": total,
        "hits": hits,
        "precision": hits / total if total else 0.0,
        "recall": hits / positives if positives else 0.0,
        "min_year_precision": min(annual) if annual else 0.0,
    }


def config_grid(states: dict[int, pd.DataFrame]) -> list[dict[str, float | int | str]]:
    grid: list[dict[str, float | int | str]] = [{"mode": "base", "lookback": 0, "start": 0.0, "span": 0.0, "floor": 0.0, "ceiling": 0.0, "rate": BASE_RATE}]
    for lookback in states:
        for start in (0.04, 0.06, 0.08):
            for span in (0.04, 0.08):
                for floor in (0.80, 0.85):
                    for ceiling in (0.98, 1.00):
                        for rate in (0.085, 0.10, 0.125):
                            grid.append({"mode": "monotonic_leg", "lookback": lookback, "start": start, "span": span, "floor": floor, "ceiling": ceiling, "rate": rate})
    return grid


def config_key(config: dict[str, float | int | str]) -> str:
    return "|".join(str(config[name]) for name in ("mode", "lookback", "start", "span", "floor", "ceiling", "rate"))


def prepare_candidates(base: pd.DataFrame, states: dict[int, pd.DataFrame], grid: list[dict[str, float | int | str]]) -> dict[str, pd.DataFrame]:
    """Build each causal score once; later folds only slice historical rows."""
    cache: dict[str, pd.DataFrame] = {}
    transformed: dict[tuple[float | int, ...], pd.DataFrame] = {}
    for config in grid:
        if config["mode"] == "base":
            candidate = base.copy()
        else:
            transform_key = (config["lookback"], config["start"], config["span"], config["floor"], config["ceiling"])
            if transform_key not in transformed:
                transformed[transform_key] = adjusted(
                    states[int(config["lookback"])],
                    float(config["start"]),
                    float(config["span"]),
                    float(config["floor"]),
                    float(config["ceiling"]),
                )
            candidate = transformed[transform_key].copy()
        candidate["selected_signal"] = signal(candidate, float(config["rate"]))
        cache[config_key(config)] = candidate.set_index("date")
    return cache


def choose(history: pd.DataFrame, grid: list[dict[str, float | int | str]], cache: dict[str, pd.DataFrame]) -> dict[str, float | int | str]:
    rows = []
    for config in grid:
        candidate = cache[config_key(config)].loc[history["date"]].reset_index()
        chosen = candidate["selected_signal"]
        stat = metrics(candidate, chosen)
        added = int((chosen & ~candidate["base_signal"].to_numpy()).sum()) if "base_signal" in candidate else 0
        rows.append({**config, **stat, "added_signals": added})
    table = pd.DataFrame(rows)
    # Require a non-base rule to have enough historical added signals.  Sparse
    # exact fits are rejected before ranking.
    table["eligible"] = (table["mode"] == "base") | (table["added_signals"] >= 5)
    table = table.loc[table["eligible"]]
    return table.sort_values(["precision", "min_year_precision", "hits", "recall"], ascending=False).iloc[0].to_dict()


def main() -> None:
    base = pd.read_csv(OUT / "severity_score_predictions_2013_2025.csv", encoding="utf-8-sig")
    base["date"] = pd.to_datetime(base["date"])
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"])
    base = base.merge(raw[["date", "close"]], on="date", how="left", validate="one_to_one")
    base["base_signal"] = (base["rank"] >= 1 - BASE_RATE) & (base["W_RSI6"] <= 45)
    states = {lookback: add_leg_state(base, lookback) for lookback in (3, 5, 7, 10)}
    grid = config_grid(states)
    cache = prepare_candidates(base, states, grid)

    summaries = []
    choices = []
    for years in HISTORY_WINDOWS:
        predictions = []
        for year in PREDICTION_YEARS:
            first = int(base["year"].min()) if years == 999 else max(int(base["year"].min()), year - years)
            history = base.loc[(base["year"] >= first) & (base["year"] < year)].copy()
            config = choose(history, grid, cache)
            test_dates = base.loc[base["year"] == year, "date"]
            test = cache[config_key(config)].loc[test_dates].reset_index().copy()
            chosen = test["selected_signal"]
            test["history_window"] = "all" if years == 999 else years
            for key, value in config.items():
                test[f"selected_{key}"] = value
            predictions.append(test)
            choices.append({"window": "all" if years == 999 else years, "year": year, "history_start": first, "history_end": year - 1, **config, **metrics(test, chosen)})
        merged = pd.concat(predictions, ignore_index=True)
        chosen = merged["selected_signal"]
        total = metrics(merged, chosen)
        total["window"] = "all" if years == 999 else years
        choice_frame = pd.DataFrame(choices)
        total["non_base_years"] = int((choice_frame.loc[choice_frame["window"] == total["window"], "mode"] != "base").sum())
        summaries.append(total)
        merged.to_csv(OUT / f"monotonic_leg_predictions_{years}.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(summaries).sort_values(["precision", "min_year_precision", "recall"], ascending=False)
    pd.DataFrame(choices).to_csv(OUT / "monotonic_leg_all_choices.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "monotonic_leg_window_summary.csv", index=False, encoding="utf-8-sig")
    (OUT / "monotonic_leg_window_summary.json").write_text(json.dumps({"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "rows": summary.to_dict(orient="records")}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
