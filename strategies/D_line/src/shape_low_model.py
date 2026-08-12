from __future__ import annotations

"""Test causal descent-shape features for five-day low-point scoring."""

import json
from datetime import datetime

import numpy as np
import pandas as pd

import rolling_extrema5_online as online
from ablate_and_ensemble_dividend_extrema5 import groups
from fit_dividend_extrema5 import OUT, PROJECT, SYMBOL, build_features


ALL_YEARS = range(2013, 2026)
TUNE_YEARS = range(2013, 2020)
HOLDOUT_YEARS = range(2020, 2026)
RATES = (0.055, 0.07, 0.085, 0.10, 0.125, 0.15)
VARIANTS = ("hgb_leaf15_l2_4", "hgb_leaf31_l2_4", "lgb_leaf15_min20", "lgb_leaf31_min20")


def shape_features(raw: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    indexed = raw.set_index("date").reindex(pd.to_datetime(labels["date"])).reset_index(drop=True)
    close = indexed["close"].astype(float)
    ret = close.pct_change()
    frame = pd.DataFrame(index=labels.index)
    for days in (3, 5, 10, 20):
        earlier_low = close.shift(1).rolling(days, min_periods=days).min()
        frame[f"new_low_{days}"] = (close < earlier_low).astype(float)
        frame[f"break_low_{days}"] = (close / earlier_low - 1.0).clip(-0.30, 0.0).fillna(0.0)
    falling = ret.lt(0)
    frame["down_streak"] = falling.groupby((~falling).cumsum()).cumsum().clip(0, 15)
    for days in (3, 5, 10):
        frame[f"negative_days_{days}"] = falling.rolling(days, min_periods=days).sum().fillna(0.0)
        frame[f"drop_speed_{days}"] = ret.rolling(days, min_periods=days).mean().fillna(0.0)
    intraday_span = (indexed["high"].astype(float) - indexed["low"].astype(float)).replace(0, np.nan)
    frame["close_in_day_range"] = ((close - indexed["low"].astype(float)) / intraday_span).fillna(0.5)
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def evaluate(frame: pd.DataFrame, stage: str, name: str) -> pd.DataFrame:
    rows = []
    for filter_name, allowed in (("none", pd.Series(True, index=frame.index)), ("week_rsi45", frame["W_RSI6"] <= 45)):
        for rate in RATES:
            selected = (frame["rank"] >= 1 - rate) & allowed
            hits = int(frame.loc[selected, "actual"].sum())
            total = int(selected.sum())
            positives = int(frame["actual"].sum())
            annual = []
            for _, part in frame.groupby("year"):
                mask = selected.loc[part.index]
                annual.append(float(part.loc[mask, "actual"].mean()) if mask.any() else 0.0)
            precision = hits / total if total else 0.0
            recall = hits / positives if positives else 0.0
            rows.append({"stage": stage, "candidate": name, "filter": filter_name, "rate": rate, "signals": total, "hits": hits, "precision": precision, "recall": recall, "min_year_precision": min(annual), "mean_year_precision": float(np.mean(annual))})
    return pd.DataFrame(rows)


def main() -> None:
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.dropna(subset=["date", "close", "high", "low"]).sort_values("date").reset_index(drop=True)
    features, labels = build_features(raw)
    custom = shape_features(raw, labels)
    all_features = pd.concat([features, custom], axis=1)
    online.FEATURE_GROUPS = {"shape_all": all_features.columns.tolist()}
    prediction_tables = []
    tuning = []
    cache = {}
    for variant in VARIANTS:
        prediction = online.staged_predictions(all_features, labels, "low", online.FEATURE_GROUPS["shape_all"], variant, ALL_YEARS)
        prediction = prediction.merge(pd.DataFrame({"date": pd.to_datetime(labels["date"]), "W_RSI6": features["W_RSI6"]}), on="date", how="left", validate="one_to_one")
        cache[variant] = prediction
        tuning.append(evaluate(prediction.loc[prediction["year"].isin(TUNE_YEARS)], "tune_2013_2019", variant))
    tuning_table = pd.concat(tuning, ignore_index=True)
    eligible = tuning_table.loc[(tuning_table["signals"] >= 35) & (tuning_table["min_year_precision"] > 0)].copy()
    choice = eligible.sort_values(["precision", "min_year_precision", "recall"], ascending=False).iloc[0]
    chosen = cache[str(choice["candidate"])]
    holdout = evaluate(chosen.loc[chosen["year"].isin(HOLDOUT_YEARS)], "holdout_2020_2025", str(choice["candidate"]))
    held = holdout.loc[(holdout["filter"] == choice["filter"]) & (holdout["rate"] == choice["rate"])].iloc[0]
    result = {
        "candidate": str(choice["candidate"]), "filter": str(choice["filter"]), "rate": float(choice["rate"]),
        "tune_precision": float(choice["precision"]), "holdout_precision": float(held["precision"]),
        "holdout_recall": float(held["recall"]), "holdout_signals": int(held["signals"]),
        "baseline_precision": 25 / 72, "precision_change": float(held["precision"] - 25 / 72),
        "min_year_precision": float(held["min_year_precision"]),
        "validated": bool(held["precision"] > 25 / 72 and held["min_year_precision"] > 0),
    }
    tuning_table.to_csv(OUT / "shape_low_tuning.csv", index=False, encoding="utf-8-sig")
    holdout.to_csv(OUT / "shape_low_holdout.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([result]).to_csv(OUT / "shape_low_selected.csv", index=False, encoding="utf-8-sig")
    chosen.to_csv(OUT / "shape_low_predictions_2013_2025.csv", index=False, encoding="utf-8-sig")
    (OUT / "shape_low_summary.json").write_text(json.dumps({"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
