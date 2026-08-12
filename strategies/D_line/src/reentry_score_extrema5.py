from __future__ import annotations

"""Test a causal deeper-reentry boost after a failed earlier low candidate."""

import json
from datetime import datetime

import numpy as np
import pandas as pd

import rolling_extrema5_online as online
from fit_dividend_extrema5 import OUT, PROJECT, SYMBOL


TUNE_YEARS = range(2013, 2020)
HOLDOUT_YEARS = range(2020, 2026)
RATES = (0.07, 0.085, 0.10, 0.125)


def with_reentry_state(frame: pd.DataFrame, lookback: int) -> pd.DataFrame:
    result = frame.sort_values("date").copy().reset_index(drop=True)
    prior_price = np.full(len(result), np.nan)
    latest_idx = -10_000
    latest_price = np.nan
    for index, row in result.iterrows():
        if index - latest_idx <= lookback:
            prior_price[index] = latest_price
        if row["base_rank"] >= 0.85:
            latest_idx = index
            latest_price = float(row["close"])
    result["prior_candidate_price"] = prior_price
    result["reentry_drop"] = result["close"] / result["prior_candidate_price"] - 1
    return result


def ranked(frame: pd.DataFrame, threshold: float, boost: float) -> pd.DataFrame:
    rows = []
    for year in sorted(frame["year"].unique()):
        calibration = frame.loc[frame["year"] == year - 1]
        test = frame.loc[frame["year"] == year]
        if calibration.empty or test.empty:
            continue
        cal_score = calibration["score"].to_numpy() + np.where(calibration["reentry_drop"] <= -threshold, boost, 0.0)
        test_score = test["score"].to_numpy() + np.where(test["reentry_drop"] <= -threshold, boost, 0.0)
        part = test[["date", "year", "actual", "base_rank", "severity", "W_RSI6", "reentry_drop"]].copy()
        part["score"] = test_score
        part["rank"] = online.relative_rank(test_score, cal_score)
        part["threshold"] = threshold
        part["boost"] = boost
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def evaluate(frame: pd.DataFrame, stage: str) -> pd.DataFrame:
    rows = []
    for rate in RATES:
        selected = frame.loc[(frame["rank"] >= 1 - rate) & (frame["W_RSI6"] <= 45)]
        signals = len(selected); hits = int(selected["actual"].sum()); positives = int(frame["actual"].sum())
        precision = hits / signals if signals else 0.0; recall = hits / positives if positives else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        annual = []
        for _, part in frame.groupby("year"):
            signal = (part["rank"] >= 1 - rate) & (part["W_RSI6"] <= 45)
            annual.append(float(part.loc[signal, "actual"].mean()) if signal.any() else 0.0)
        rows.append({"stage": stage, "lookback": int(frame["lookback"].iloc[0]), "threshold": float(frame["threshold"].iloc[0]), "boost": float(frame["boost"].iloc[0]), "rate": rate, "signals": signals, "hits": hits, "precision": precision, "recall": recall, "f1": f1, "min_year_precision": min(annual)})
    return pd.DataFrame(rows)


def main() -> None:
    base = pd.read_csv(OUT / "severity_score_predictions_2013_2025.csv", encoding="utf-8-sig")
    base["date"] = pd.to_datetime(base["date"])
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"])
    base = base.merge(raw[["date", "close"]], on="date", how="left", validate="one_to_one")
    cache = {}
    tuning_tables = []
    for lookback in (5, 10, 15):
        state = with_reentry_state(base, lookback)
        for threshold in (0.04, 0.06, 0.08):
            for boost in (0.05, 0.10, 0.15, 0.20):
                prediction = ranked(state, threshold, boost)
                prediction["lookback"] = lookback
                cache[(lookback, threshold, boost)] = prediction
                tuning_tables.append(evaluate(prediction.loc[prediction["year"].isin(TUNE_YEARS)], "tune_2013_2019"))
    tuning = pd.concat(tuning_tables, ignore_index=True)
    eligible = tuning.loc[(tuning["signals"] >= len(TUNE_YEARS) * 5) & (tuning["min_year_precision"] > 0)].copy()
    choice = eligible.sort_values(["precision", "min_year_precision", "recall", "f1"], ascending=False).iloc[0]
    chosen = cache[(int(choice["lookback"]), float(choice["threshold"]), float(choice["boost"]))]
    held = evaluate(chosen.loc[chosen["year"].isin(HOLDOUT_YEARS)], "holdout_2020_2025")
    valid = held.loc[held["rate"] == float(choice["rate"])].iloc[0]
    baseline = 27 / 81
    result = {
        "lookback": int(choice["lookback"]),
        "threshold": float(choice["threshold"]),
        "boost": float(choice["boost"]),
        "rate": float(choice["rate"]),
        "tune_precision": float(choice["precision"]),
        "tune_recall": float(choice["recall"]),
        "tune_f1": float(choice["f1"]),
        "holdout_precision": float(valid["precision"]),
        "holdout_recall": float(valid["recall"]),
        "holdout_f1": float(valid["f1"]),
        "holdout_signals": int(valid["signals"]),
        "baseline_precision": baseline,
        "precision_change": float(valid["precision"] - baseline),
        "min_year_precision": float(valid["min_year_precision"]),
        "validated": bool(valid["precision"] > baseline and valid["min_year_precision"] > 0),
    }
    tuning.to_csv(OUT / "reentry_score_tuning.csv", index=False, encoding="utf-8-sig")
    held.to_csv(OUT / "reentry_score_holdout.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([result]).to_csv(OUT / "reentry_score_selected.csv", index=False, encoding="utf-8-sig")
    chosen.to_csv(OUT / "reentry_score_predictions_2013_2025.csv", index=False, encoding="utf-8-sig")
    (OUT / "reentry_score_summary.json").write_text(json.dumps({"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
