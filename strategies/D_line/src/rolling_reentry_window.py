from __future__ import annotations

"""Select a second-entry rule yearly with only the chosen historical window."""

import json
from datetime import datetime

import numpy as np
import pandas as pd

from fit_dividend_extrema5 import OUT, PROJECT, SYMBOL
from reentry_addon_extrema5 import BASE_RATE, add_reentry_state


PREDICTION_YEARS = range(2020, 2026)
WINDOWS = (3, 5, 7, 999)


def mask_for(frame: pd.DataFrame, drop: float, rsi: float, severity: float) -> pd.Series:
    return (
        ~frame["base_signal"]
        & (frame["reentry_drop"] <= -drop)
        & (frame["W_RSI6"] <= rsi)
        & (frame["severity"] >= severity)
    )


def metrics(frame: pd.DataFrame, signal: pd.Series) -> dict[str, float | int]:
    picks = frame.loc[signal]
    signals = int(signal.sum())
    hits = int(picks["actual"].sum())
    positives = int(frame["actual"].sum())
    precision = hits / signals if signals else 0.0
    recall = hits / positives if positives else 0.0
    annual = []
    for _, part in frame.groupby("year"):
        chosen = signal.loc[part.index]
        annual.append(float(part.loc[chosen, "actual"].mean()) if chosen.any() else 0.0)
    return {
        "signals": signals,
        "hits": hits,
        "precision": precision,
        "recall": recall,
        "min_year_precision": min(annual) if annual else 0.0,
    }


def candidate_grid(states: dict[int, pd.DataFrame]) -> list[dict[str, float | int | str]]:
    grid: list[dict[str, float | int | str]] = [{"mode": "baseline", "lookback": 0, "drop": 0.0, "week_rsi_max": 0.0, "severity_min": 0.0}]
    for lookback, frame in states.items():
        for drop in (0.04, 0.06, 0.08, 0.10, 0.12):
            for rsi in (20, 25, 30, 35):
                for severity in (0.80, 0.90, 0.95):
                    grid.append({"mode": "reentry", "lookback": lookback, "drop": drop, "week_rsi_max": rsi, "severity_min": severity})
    return grid


def signal_for(frame: pd.DataFrame, candidate: dict[str, float | int | str]) -> tuple[pd.Series, pd.Series]:
    base = frame["base_signal"].copy()
    if candidate["mode"] == "baseline":
        return base, pd.Series(False, index=frame.index)
    addon = mask_for(frame, float(candidate["drop"]), float(candidate["week_rsi_max"]), float(candidate["severity_min"]))
    return base | addon, addon


def select_candidate(history: pd.DataFrame, states: dict[int, pd.DataFrame], candidates: list[dict[str, float | int | str]]) -> dict[str, float | int | str]:
    rows = []
    for candidate in candidates:
        source = history if candidate["mode"] == "baseline" else states[int(candidate["lookback"])] .loc[history.index]
        signal, addon = signal_for(source, candidate)
        stat = metrics(source, signal)
        addon_signals = int(addon.sum())
        # A candidate must improve on baseline using at least five historical
        # second-entry events; otherwise it is too sparse to be trusted.
        eligible = candidate["mode"] == "baseline" or addon_signals >= 5
        rows.append({**candidate, **stat, "addon_signals": addon_signals, "eligible": eligible})
    table = pd.DataFrame(rows)
    table = table.loc[table["eligible"]].copy()
    return table.sort_values(["precision", "min_year_precision", "hits", "recall"], ascending=False).iloc[0].to_dict()


def run_window(frame: pd.DataFrame, states: dict[int, pd.DataFrame], candidates: list[dict[str, float | int | str]], window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    choices = []
    predictions = []
    for year in PREDICTION_YEARS:
        start = max(int(frame["year"].min()), year - window) if window != 999 else int(frame["year"].min())
        history = frame.loc[(frame["year"] >= start) & (frame["year"] < year)]
        choice = select_candidate(history, states, candidates)
        source = frame.loc[frame["year"] == year] if choice["mode"] == "baseline" else states[int(choice["lookback"])] .loc[frame["year"] == year]
        signal, addon = signal_for(source, choice)
        part = source.copy()
        part["selected_signal"] = signal
        part["addon_signal"] = addon
        part["window_years"] = "all" if window == 999 else window
        for key, value in choice.items():
            part[f"selected_{key}"] = value
        predictions.append(part)
        choices.append({"year": year, "history_start": start, "history_end": year - 1, "window_years": "all" if window == 999 else window, **choice, **metrics(source, signal), "addon_signals_test": int(addon.sum()), "addon_hits_test": int(source.loc[addon, "actual"].sum())})
    return pd.DataFrame(choices), pd.concat(predictions, ignore_index=True)


def summary(predictions: pd.DataFrame, window: int) -> dict[str, float | int | str]:
    signal = predictions["selected_signal"]
    stat = metrics(predictions, signal)
    stat["window_years"] = "all" if window == 999 else window
    stat["addon_signals"] = int(predictions["addon_signal"].sum())
    stat["addon_hits"] = int(predictions.loc[predictions["addon_signal"], "actual"].sum())
    return stat


def main() -> None:
    base = pd.read_csv(OUT / "severity_score_predictions_2013_2025.csv", encoding="utf-8-sig")
    base["date"] = pd.to_datetime(base["date"])
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"])
    base = base.merge(raw[["date", "close"]], on="date", how="left", validate="one_to_one")
    base["base_signal"] = (base["rank"] >= 1 - BASE_RATE) & (base["W_RSI6"] <= 45)
    states = {lookback: add_reentry_state(base, lookback) for lookback in (3, 5, 7, 10)}
    candidates = candidate_grid(states)

    all_choices = []
    all_summaries = []
    for window in WINDOWS:
        choices, predictions = run_window(base, states, candidates, window)
        choices.to_csv(OUT / f"rolling_reentry_choices_{window}.csv", index=False, encoding="utf-8-sig")
        predictions.to_csv(OUT / f"rolling_reentry_predictions_{window}.csv", index=False, encoding="utf-8-sig")
        all_choices.append(choices)
        all_summaries.append(summary(predictions, window))
    choice_table = pd.concat(all_choices, ignore_index=True)
    result = pd.DataFrame(all_summaries).sort_values(["precision", "min_year_precision", "recall"], ascending=False)
    result.to_csv(OUT / "rolling_reentry_window_summary.csv", index=False, encoding="utf-8-sig")
    choice_table.to_csv(OUT / "rolling_reentry_all_choices.csv", index=False, encoding="utf-8-sig")
    (OUT / "rolling_reentry_window_summary.json").write_text(json.dumps({"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "windows": result.to_dict(orient="records")}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
