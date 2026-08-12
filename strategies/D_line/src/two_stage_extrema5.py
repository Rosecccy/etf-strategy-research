from __future__ import annotations

"""Strict secondary calibration for the online five-day extrema model.

The first stage generates an online score exactly as before.  A compact second
stage learns, only from already-realised first-stage predictions, when that
score is credible given daily/weekly trend and volume context.  It follows the
same train-through-Y-2 / calibrate-on-Y-1 / test-on-Y timing as the base model.
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
from filter_extrema5_audit import HIGH_CANDIDATE, LOW_CANDIDATE, candidate_predictions, with_feature_rows
from fit_dividend_extrema5 import OUT, PROJECT, SYMBOL, WINDOW, build_features, scores


TUNE_YEARS = range(2016, 2020)
HOLDOUT_YEARS = range(2020, 2026)
ALL_YEARS = range(2013, 2026)
RATES = (0.055, 0.07, 0.085, 0.10, 0.125, 0.15)

BASE_COLUMNS = ["score", "rank"]
STATE_COLUMNS = BASE_COLUMNS + [
    "D_RSI6", "D_KDJ_J", "D_DIF", "D_LON", "W_RSI6", "W_KDJ_J",
    "ret_1", "ret_5", "ret_20", "ma_ratio_20", "ma_ratio_60",
    "range_pos_20", "drawdown_20", "realized_vol_20", "volume_ratio_20", "intraday_range",
]


def make_model(kind: str, positive_rate: float) -> object:
    if kind == "logit":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.04, class_weight="balanced", max_iter=3000)),
        ])
    if kind == "hgb":
        weight = max((1 - positive_rate) / max(positive_rate, 0.001), 1.0)
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.04, max_leaf_nodes=7, min_samples_leaf=35,
                l2_regularization=5.0, class_weight={0: 1.0, 1: weight}, random_state=19,
            )),
        ])
    raise ValueError(kind)


def staged_secondary(frame: pd.DataFrame, columns: list[str], kind: str, years: range) -> pd.DataFrame:
    rows = []
    for year in years:
        train = frame.loc[frame["year"] <= year - 2].copy()
        calibration = frame.loc[frame["year"] == year - 1].copy()
        test = frame.loc[frame["year"] == year].copy()
        if len(train) <= WINDOW:
            continue
        train = train.iloc[:-WINDOW].copy()
        if train["actual"].sum() < 20 or calibration.empty or test.empty:
            continue
        model = make_model(kind, float(train["actual"].mean()))
        model.fit(train[columns], train["actual"])
        calibration_score = scores(model, calibration[columns])
        test_score = scores(model, test[columns])
        ranks = online.relative_rank(test_score, calibration_score)
        rows.append(pd.DataFrame({
            "date": test["date"].to_numpy(), "year": year, "actual": test["actual"].to_numpy(),
            "score": test_score, "rank": ranks, "candidate": f"two_stage_{kind}_{'state' if len(columns) > 2 else 'base'}",
        }))
    return pd.concat(rows, ignore_index=True)


def assess(predictions: pd.DataFrame, target: str, stage: str, setting: str) -> pd.DataFrame:
    rows = []
    for rate in RATES:
        total = online.threshold_metrics(predictions, rate)
        yearly = [online.threshold_metrics(part, rate) for _, part in predictions.groupby("year")]
        rows.append({
            "target": target, "stage": stage, "setting": setting, **total,
            "min_year_precision": min(item["precision"] for item in yearly),
            "mean_year_precision": float(np.mean([item["precision"] for item in yearly])),
        })
    return pd.DataFrame(rows)


def one_target(target: str, raw_features: pd.DataFrame, labels: pd.DataFrame) -> tuple[dict[str, object], list[pd.DataFrame]]:
    candidate = LOW_CANDIDATE if target == "low" else HIGH_CANDIDATE
    base = candidate_predictions(raw_features, labels, target, candidate, ALL_YEARS)
    frame = with_feature_rows(base, raw_features, labels)
    tables = []
    prediction_cache: dict[str, pd.DataFrame] = {}
    for kind in ("logit", "hgb"):
        for feature_name, columns in (("base", BASE_COLUMNS), ("state", STATE_COLUMNS)):
            setting = f"{kind}:{feature_name}"
            prediction = staged_secondary(frame, columns, kind, ALL_YEARS)
            prediction_cache[setting] = prediction
            tables.append(assess(prediction.loc[prediction["year"].isin(TUNE_YEARS)], target, "tune_2016_2019", setting))
    tuning = pd.concat(tables, ignore_index=True)
    # A single sharp best threshold is usually a sign of selection noise.  Rank
    # each setting by the worst precision at its immediately adjacent rates.
    tuning["neighbor_precision_floor"] = 0.0
    for setting, part in tuning.groupby("setting"):
        ordered = part.sort_values("rate")
        floors = []
        values = ordered["precision"].to_list()
        for index in range(len(values)):
            floors.append(min(values[max(0, index - 1) : min(len(values), index + 2)]))
        tuning.loc[ordered.index, "neighbor_precision_floor"] = floors
    eligible = tuning.loc[(tuning["signals"] >= len(TUNE_YEARS) * 5) & (tuning["min_year_precision"] > 0)].copy()
    choice = eligible.sort_values(
        ["neighbor_precision_floor", "precision", "min_year_precision", "recall", "f1"], ascending=False
    ).iloc[0]
    chosen_prediction = prediction_cache[str(choice["setting"])]
    holdout = chosen_prediction.loc[chosen_prediction["year"].isin(HOLDOUT_YEARS)]
    held = assess(holdout, target, "holdout_2020_2025", str(choice["setting"]))
    # Keep every held-out setting in the audit so reviewers can see whether the
    # neighborhood rule was genuinely protective rather than cherry-picked.
    held_all = [held]
    for setting, prediction in prediction_cache.items():
        if setting == str(choice["setting"]):
            continue
        held_all.append(assess(prediction.loc[prediction["year"].isin(HOLDOUT_YEARS)], target, "holdout_2020_2025", setting))
    validate = held.loc[held["rate"] == float(choice["rate"])].iloc[0]
    baseline = online.threshold_metrics(base.loc[base["year"].isin(HOLDOUT_YEARS)], 0.15)
    result = {
        "target": target,
        "base_candidate": candidate,
        "selected_second_stage": str(choice["setting"]),
        "selected_rate": float(choice["rate"]),
        "neighbor_precision_floor": float(choice["neighbor_precision_floor"]),
        "tune_precision": float(choice["precision"]),
        "tune_recall": float(choice["recall"]),
        "tune_f1": float(choice["f1"]),
        "holdout_precision": float(validate["precision"]),
        "holdout_recall": float(validate["recall"]),
        "holdout_f1": float(validate["f1"]),
        "holdout_signals": int(validate["signals"]),
        "baseline_precision_rate15": float(baseline["precision"]),
        "holdout_precision_change": float(validate["precision"] - baseline["precision"]),
        "holdout_min_year_precision": float(validate["min_year_precision"]),
        "validated": bool(validate["precision"] > baseline["precision"] and validate["min_year_precision"] > 0),
    }
    return result, [tuning, *held_all]


def main() -> None:
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    features, labels = build_features(raw)
    online.FEATURE_GROUPS = groups(features)
    results = []
    tables = []
    for target in ("low", "high"):
        result, target_tables = one_target(target, features, labels)
        results.append(result)
        tables.extend(target_tables)
    result_frame = pd.DataFrame(results)
    table = pd.concat(tables, ignore_index=True)
    result_frame.to_csv(OUT / "two_stage_selected.csv", index=False, encoding="utf-8-sig")
    table.to_csv(OUT / "two_stage_ranking.csv", index=False, encoding="utf-8-sig")
    payload = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "results": results}
    (OUT / "two_stage_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
