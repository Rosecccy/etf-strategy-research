from __future__ import annotations

"""Validate a causal second-entry signal after a prior low signal keeps falling."""

import json
from datetime import datetime

import numpy as np
import pandas as pd

from fit_dividend_extrema5 import OUT, PROJECT, SYMBOL


TUNE_YEARS = range(2013, 2020)
HOLDOUT_YEARS = range(2020, 2026)
BASE_RATE = 0.10
BASELINE_PRECISION = 25 / 72


def add_reentry_state(frame: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Use only earlier model signals to measure a fresh leg down."""
    result = frame.sort_values("date").copy().reset_index(drop=True)
    previous_index = -10_000
    previous_price = np.nan
    prices = np.full(len(result), np.nan)
    for index, row in result.iterrows():
        if index - previous_index <= lookback:
            prices[index] = previous_price
        if bool(row["base_signal"]):
            previous_index = index
            previous_price = float(row["close"])
    result["prior_signal_close"] = prices
    result["reentry_drop"] = result["close"] / result["prior_signal_close"] - 1.0
    return result


def score(frame: pd.DataFrame, stage: str, lookback: int, drop: float, rsi: float, severity: float) -> dict[str, float | int | str]:
    base = frame["base_signal"]
    addon = (
        ~base
        & (frame["reentry_drop"] <= -drop)
        & (frame["W_RSI6"] <= rsi)
        & (frame["severity"] >= severity)
    )
    signal = base | addon
    selected = frame.loc[signal]
    hits = int(selected["actual"].sum())
    precision = hits / len(selected) if len(selected) else 0.0
    recall = hits / int(frame["actual"].sum()) if int(frame["actual"].sum()) else 0.0
    yearly = []
    for _, part in frame.groupby("year"):
        yearly_signal = signal.loc[part.index]
        yearly.append(float(part.loc[yearly_signal, "actual"].mean()) if yearly_signal.any() else 0.0)
    return {
        "stage": stage,
        "lookback": lookback,
        "drop": drop,
        "week_rsi_max": rsi,
        "severity_min": severity,
        "signals": int(signal.sum()),
        "hits": hits,
        "addon_signals": int(addon.sum()),
        "addon_hits": int(frame.loc[addon, "actual"].sum()),
        "precision": precision,
        "recall": recall,
        "min_year_precision": min(yearly),
    }


def main() -> None:
    frame = pd.read_csv(OUT / "severity_score_predictions_2013_2025.csv", encoding="utf-8-sig")
    frame["date"] = pd.to_datetime(frame["date"])
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"])
    frame = frame.merge(raw[["date", "close"]], on="date", how="left", validate="one_to_one")
    frame["base_signal"] = (frame["rank"] >= 1 - BASE_RATE) & (frame["W_RSI6"] <= 45)

    candidates = []
    states: dict[int, pd.DataFrame] = {}
    for lookback in (3, 5, 7, 10):
        state = add_reentry_state(frame, lookback)
        states[lookback] = state
        for drop in (0.04, 0.06, 0.08, 0.10, 0.12):
            for rsi in (20, 25, 30, 35):
                for severity in (0.80, 0.90, 0.95):
                    candidates.append(score(state.loc[state["year"].isin(TUNE_YEARS)], "tune_2013_2019", lookback, drop, rsi, severity))
    tuning = pd.DataFrame(candidates)
    eligible = tuning.loc[(tuning["addon_signals"] >= 5) & (tuning["min_year_precision"] > 0)].copy()
    choice = eligible.sort_values(["precision", "min_year_precision", "addon_hits", "recall"], ascending=False).iloc[0]
    selected_state = states[int(choice["lookback"])]
    held = score(selected_state.loc[selected_state["year"].isin(HOLDOUT_YEARS)], "holdout_2020_2025", int(choice["lookback"]), float(choice["drop"]), float(choice["week_rsi_max"]), float(choice["severity_min"]))
    full = selected_state.copy()
    full["addon_signal"] = (
        ~full["base_signal"]
        & (full["reentry_drop"] <= -float(choice["drop"]))
        & (full["W_RSI6"] <= float(choice["week_rsi_max"]))
        & (full["severity"] >= float(choice["severity_min"]))
    )
    full["selected_signal"] = full["base_signal"] | full["addon_signal"]
    result = {
        "lookback": int(choice["lookback"]),
        "drop": float(choice["drop"]),
        "week_rsi_max": float(choice["week_rsi_max"]),
        "severity_min": float(choice["severity_min"]),
        "tune_precision": float(choice["precision"]),
        "holdout_precision": float(held["precision"]),
        "holdout_recall": float(held["recall"]),
        "holdout_signals": int(held["signals"]),
        "holdout_addon_signals": int(held["addon_signals"]),
        "holdout_addon_hits": int(held["addon_hits"]),
        "baseline_precision": BASELINE_PRECISION,
        "precision_change": float(held["precision"] - BASELINE_PRECISION),
        "min_year_precision": float(held["min_year_precision"]),
        "validated": bool(held["precision"] >= BASELINE_PRECISION and held["addon_signals"] >= 3 and held["min_year_precision"] > 0),
    }
    tuning.to_csv(OUT / "reentry_addon_tuning.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([held]).to_csv(OUT / "reentry_addon_holdout.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([result]).to_csv(OUT / "reentry_addon_selected.csv", index=False, encoding="utf-8-sig")
    full.to_csv(OUT / "reentry_addon_predictions_2013_2025.csv", index=False, encoding="utf-8-sig")
    (OUT / "reentry_addon_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
