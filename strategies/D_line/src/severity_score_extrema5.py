from __future__ import annotations

"""Causal severity-aware re-ranking for five-day low-point candidates.

This preserves the strict online base score, then adds only information already
known at each close: how deeply price sits below its own recent peak, how
oversold daily RSI is, and how near the 20-day range floor price is.  All ranks
are calibrated against the preceding year, never against the future test year.
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd

import rolling_extrema5_online as online
from ablate_and_ensemble_dividend_extrema5 import groups
from filter_extrema5_audit import LOW_CANDIDATE, candidate_predictions, with_feature_rows
from fit_dividend_extrema5 import OUT, PROJECT, SYMBOL, build_features


ALL_YEARS = range(2013, 2026)
TUNE_YEARS = range(2013, 2020)
HOLDOUT_YEARS = range(2020, 2026)
RATES = (0.055, 0.07, 0.085, 0.10, 0.125, 0.15)
WEIGHTS = (0.55, 0.65, 0.75, 0.85)


def severity(frame: pd.DataFrame, reference: pd.DataFrame, mode: str) -> np.ndarray:
    parts = [
        online.relative_rank(-frame["drawdown_20"].to_numpy(), -reference["drawdown_20"].to_numpy()),
        online.relative_rank(-frame["D_RSI6"].to_numpy(), -reference["D_RSI6"].to_numpy()),
    ]
    if mode == "deep3":
        parts.append(online.relative_rank(-frame["range_pos_20"].to_numpy(), -reference["range_pos_20"].to_numpy()))
    return np.mean(parts, axis=0)


def adjusted_predictions(base: pd.DataFrame, weight: float, mode: str) -> pd.DataFrame:
    rows = []
    for year in ALL_YEARS:
        calibration = base.loc[base["year"] == year - 1]
        test = base.loc[base["year"] == year]
        if calibration.empty or test.empty:
            continue
        sev_cal = severity(calibration, calibration, mode)
        sev_test = severity(test, calibration, mode)
        combined_cal = weight * calibration["rank"].to_numpy() + (1 - weight) * sev_cal
        combined_test = weight * test["rank"].to_numpy() + (1 - weight) * sev_test
        rows.append(pd.DataFrame({
            "date": test["date"].to_numpy(), "year": year, "actual": test["actual"].to_numpy(),
            "base_rank": test["rank"].to_numpy(), "severity": sev_test, "score": combined_test,
            "rank": online.relative_rank(combined_test, combined_cal), "weight": weight, "mode": mode,
            "W_RSI6": test["W_RSI6"].to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True)


def evaluate(frame: pd.DataFrame, stage: str, filter_name: str) -> pd.DataFrame:
    allowed = pd.Series(True, index=frame.index) if filter_name == "none" else frame["W_RSI6"] <= 45
    rows = []
    for rate in RATES:
        selected = frame.loc[(frame["rank"] >= 1 - rate) & allowed]
        signals = len(selected); hits = int(selected["actual"].sum()); positives = int(frame["actual"].sum())
        precision = hits / signals if signals else 0.0; recall = hits / positives if positives else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        yearly = []
        for _, part in frame.groupby("year"):
            mask = (part["rank"] >= 1 - rate) & ((pd.Series(True, index=part.index)) if filter_name == "none" else (part["W_RSI6"] <= 45))
            c = int(mask.sum()); h = int(part.loc[mask, "actual"].sum())
            yearly.append(h / c if c else 0.0)
        rows.append({
            "stage": stage, "mode": frame["mode"].iloc[0], "weight": frame["weight"].iloc[0], "filter": filter_name,
            "rate": rate, "signals": signals, "hits": hits, "precision": precision, "recall": recall, "f1": f1,
            "min_year_precision": min(yearly), "mean_year_precision": float(np.mean(yearly)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    features, labels = build_features(raw)
    online.FEATURE_GROUPS = groups(features)
    base = candidate_predictions(features, labels, "low", LOW_CANDIDATE, ALL_YEARS)
    base = with_feature_rows(base, features, labels)
    cache: dict[tuple[str, float], pd.DataFrame] = {}
    tuning_tables = []
    for mode in ("deep2", "deep3"):
        for weight in WEIGHTS:
            prediction = adjusted_predictions(base, weight, mode)
            cache[(mode, weight)] = prediction
            for filter_name in ("none", "week_rsi45"):
                tuning_tables.append(evaluate(prediction.loc[prediction["year"].isin(TUNE_YEARS)], "tune_2013_2019", filter_name))
    tuning = pd.concat(tuning_tables, ignore_index=True)
    eligible = tuning.loc[(tuning["signals"] >= len(TUNE_YEARS) * 5) & (tuning["min_year_precision"] > 0)].copy()
    choice = eligible.sort_values(["precision", "min_year_precision", "recall", "f1"], ascending=False).iloc[0]
    chosen = cache[(str(choice["mode"]), float(choice["weight"]))]
    held = evaluate(chosen.loc[chosen["year"].isin(HOLDOUT_YEARS)], "holdout_2020_2025", str(choice["filter"]))
    validated = held.loc[held["rate"] == float(choice["rate"])].iloc[0]
    baseline = online.threshold_metrics(base.loc[base["year"].isin(HOLDOUT_YEARS)], 0.15)
    result = {
        "mode": str(choice["mode"]), "base_weight": float(choice["weight"]), "filter": str(choice["filter"]), "rate": float(choice["rate"]),
        "tune_precision": float(choice["precision"]), "tune_recall": float(choice["recall"]), "tune_f1": float(choice["f1"]),
        "holdout_precision": float(validated["precision"]), "holdout_recall": float(validated["recall"]), "holdout_f1": float(validated["f1"]),
        "holdout_signals": int(validated["signals"]), "baseline_precision": float(baseline["precision"]),
        "precision_change": float(validated["precision"] - baseline["precision"]), "min_year_precision": float(validated["min_year_precision"]),
        "validated": bool(validated["precision"] > baseline["precision"] and validated["min_year_precision"] > 0),
    }
    tuning.to_csv(OUT / "severity_score_tuning.csv", index=False, encoding="utf-8-sig")
    held.to_csv(OUT / "severity_score_holdout.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([result]).to_csv(OUT / "severity_score_selected.csv", index=False, encoding="utf-8-sig")
    chosen.to_csv(OUT / "severity_score_predictions_2013_2025.csv", index=False, encoding="utf-8-sig")
    (OUT / "severity_score_summary.json").write_text(json.dumps({"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
