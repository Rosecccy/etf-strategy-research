from __future__ import annotations

"""Add causal rebound-quality ranking to the strict five-day low-point model.

The secondary learner sees only same-day features and the already-online base
score.  Its historical target distinguishes a tradable rebound from a hard
negative: after the day, price must rebound within five sessions without first
falling materially further.  Hard negatives that the base model had rated highly
receive extra training weight.
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import rolling_extrema5_online as online
from ablate_and_ensemble_dividend_extrema5 import groups
from filter_extrema5_audit import LOW_CANDIDATE, candidate_predictions, with_feature_rows
from fit_dividend_extrema5 import OUT, PROJECT, SYMBOL, build_features, scores


ALL_YEARS = range(2013, 2026)
TUNE_YEARS = range(2016, 2020)
HOLDOUT_YEARS = range(2020, 2026)
RATES = (0.055, 0.07, 0.085, 0.10, 0.125)
QUALITY_RULES = ((0.02, -0.01), (0.03, -0.02), (0.04, -0.03))
FEATURES = [
    "rank", "score", "D_RSI6", "D_KDJ_J", "D_DIF", "D_LON", "W_RSI6", "W_KDJ_J",
    "ret_1", "ret_5", "ret_20", "ma_ratio_20", "ma_ratio_60", "range_pos_20",
    "drawdown_20", "realized_vol_20", "volume_ratio_20", "intraday_range",
]


def future_quality(raw: pd.DataFrame, up: float, safety: float) -> pd.DataFrame:
    close = raw["close"].astype(float).reset_index(drop=True)
    max5 = pd.concat([close.shift(-step) for step in range(1, 6)], axis=1).max(axis=1) / close - 1
    min3 = pd.concat([close.shift(-step) for step in range(1, 4)], axis=1).min(axis=1) / close - 1
    return pd.DataFrame({"date": raw["date"], "quality": ((max5 >= up) & (min3 >= safety)).astype(int), "max5": max5, "min3": min3})


def model_for(kind: str, positive_rate: float) -> Pipeline:
    if kind == "logit":
        estimator = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000)
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", estimator)])
    weight = max((1 - positive_rate) / max(positive_rate, 0.001), 1.0)
    estimator = HistGradientBoostingClassifier(
        learning_rate=0.04, max_leaf_nodes=7, min_samples_leaf=40, l2_regularization=6.0,
        class_weight={0: 1.0, 1: weight}, random_state=31,
    )
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", estimator)])


def staged_quality(frame: pd.DataFrame, rule: tuple[float, float], kind: str) -> pd.DataFrame:
    up, safety = rule
    rows = []
    for year in ALL_YEARS:
        train = frame.loc[frame["year"] <= year - 2].copy()
        calibration = frame.loc[frame["year"] == year - 1].copy()
        test = frame.loc[frame["year"] == year].copy()
        if len(train) < 400 or train["quality"].sum() < 30 or calibration.empty or test.empty:
            continue
        # A high initial low-point score followed by continued decline is a
        # costly false signal, so it is deliberately a harder negative.
        weight = np.where((train["rank"] >= 0.70) & (train["quality"] == 0), 3.0, 1.0)
        model = model_for(kind, float(train["quality"].mean()))
        model.fit(train[FEATURES], train["quality"], model__sample_weight=weight)
        calibration_score = scores(model, calibration[FEATURES])
        test_score = scores(model, test[FEATURES])
        # Preserve information about the original extremum model while allowing
        # rebound quality to reorder its close calls.
        combined = 0.45 * test["rank"].to_numpy() + 0.55 * online.relative_rank(test_score, calibration_score)
        calibration_combined = 0.45 * calibration["rank"].to_numpy() + 0.55 * online.relative_rank(calibration_score, calibration_score)
        rows.append(pd.DataFrame({
            "date": test["date"].to_numpy(), "year": year, "actual": test["actual"].to_numpy(),
            "quality": test["quality"].to_numpy(), "score": combined,
            "rank": online.relative_rank(combined, calibration_combined),
            "rule": f"up{up:.0%}_safe{safety:.0%}", "kind": kind,
        }))
    return pd.concat(rows, ignore_index=True)


def assess(frame: pd.DataFrame, stage: str) -> pd.DataFrame:
    rows = []
    for rate in RATES:
        total = online.threshold_metrics(frame, rate)
        years = [online.threshold_metrics(part, rate) for _, part in frame.groupby("year")]
        rows.append({
            "stage": stage, "rule": frame["rule"].iloc[0], "kind": frame["kind"].iloc[0], **total,
            "min_year_precision": min(item["precision"] for item in years),
            "mean_year_precision": float(np.mean([item["precision"] for item in years])),
        })
    return pd.DataFrame(rows)


def main() -> None:
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    features, labels = build_features(raw)
    online.FEATURE_GROUPS = groups(features)
    base = candidate_predictions(features, labels, "low", LOW_CANDIDATE, ALL_YEARS)
    frame = with_feature_rows(base, features, labels)
    prediction_cache: dict[tuple[tuple[float, float], str], pd.DataFrame] = {}
    tables = []
    for rule in QUALITY_RULES:
        quality = future_quality(raw, *rule)
        prepared = frame.merge(quality, on="date", how="left", validate="one_to_one")
        prepared = prepared.dropna(subset=FEATURES + ["quality"]).copy()
        for kind in ("logit", "hgb"):
            prediction = staged_quality(prepared, rule, kind)
            prediction_cache[(rule, kind)] = prediction
            tables.append(assess(prediction.loc[prediction["year"].isin(TUNE_YEARS)], "tune_2016_2019"))
    tuning = pd.concat(tables, ignore_index=True)
    eligible = tuning.loc[(tuning["signals"] >= len(TUNE_YEARS) * 5) & (tuning["min_year_precision"] > 0)].copy()
    choice = eligible.sort_values(["precision", "min_year_precision", "recall", "f1"], ascending=False).iloc[0]
    up = float(str(choice["rule"]).split("_")[0].removeprefix("up").replace("%", "")) / 100
    safety = float(str(choice["rule"]).split("safe")[1].replace("%", "")) / 100
    chosen = prediction_cache[((up, safety), str(choice["kind"]))]
    holdout = chosen.loc[chosen["year"].isin(HOLDOUT_YEARS)]
    held = assess(holdout, "holdout_2020_2025")
    selected_hold = held.loc[held["rate"] == float(choice["rate"])].iloc[0]
    baseline = online.threshold_metrics(base.loc[base["year"].isin(HOLDOUT_YEARS)], 0.15)
    result = {
        "base_candidate": LOW_CANDIDATE, "rule": str(choice["rule"]), "kind": str(choice["kind"]),
        "rate": float(choice["rate"]), "tune_precision": float(choice["precision"]),
        "tune_recall": float(choice["recall"]), "tune_f1": float(choice["f1"]),
        "holdout_precision": float(selected_hold["precision"]), "holdout_recall": float(selected_hold["recall"]),
        "holdout_f1": float(selected_hold["f1"]), "holdout_signals": int(selected_hold["signals"]),
        "baseline_precision": float(baseline["precision"]),
        "precision_change": float(selected_hold["precision"] - baseline["precision"]),
        "min_year_precision": float(selected_hold["min_year_precision"]),
        "validated": bool(selected_hold["precision"] > baseline["precision"] and selected_hold["min_year_precision"] > 0),
    }
    tuning.to_csv(OUT / "quality_rank_tuning.csv", index=False, encoding="utf-8-sig")
    held.to_csv(OUT / "quality_rank_holdout.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([result]).to_csv(OUT / "quality_rank_selected.csv", index=False, encoding="utf-8-sig")
    chosen.to_csv(OUT / "quality_rank_predictions_2013_2025.csv", index=False, encoding="utf-8-sig")
    (OUT / "quality_rank_summary.json").write_text(json.dumps({"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
