from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from fit_dividend_extrema5 import (
    FIRST_OOS_YEAR,
    LAST_COMPLETE_OOS_YEAR,
    OUT,
    PROJECT,
    SYMBOL,
    build_features,
    event_metrics,
    scores,
)


TUNE_LAST_YEAR = 2019
HOLDOUT_FIRST_YEAR = 2020
VARIANTS = [
    ("rf_d5_l3", "rf", 5, 3),
    ("rf_d5_l6", "rf", 5, 6),
    ("rf_d7_l3", "rf", 7, 3),
    ("rf_d7_l6", "rf", 7, 6),
    ("rf_d10_l3", "rf", 10, 3),
    ("rf_d10_l6", "rf", 10, 6),
    ("extra_d6_l3", "extra", 6, 3),
    ("extra_d8_l3", "extra", 8, 3),
    ("extra_d10_l5", "extra", 10, 5),
    ("hgb_leaf7_l2_1", "hgb", 7, 1),
    ("hgb_leaf15_l2_1", "hgb", 15, 1),
    ("hgb_leaf31_l2_1", "hgb", 31, 1),
    ("hgb_leaf15_l2_4", "hgb", 15, 4),
    ("hgb_leaf31_l2_4", "hgb", 31, 4),
    ("lgb_leaf7_min20", "lgb", 7, 20),
    ("lgb_leaf7_min45", "lgb", 7, 45),
    ("lgb_leaf15_min20", "lgb", 15, 20),
    ("lgb_leaf15_min45", "lgb", 15, 45),
    ("lgb_leaf31_min20", "lgb", 31, 20),
    ("lgb_leaf31_min45", "lgb", 31, 45),
]


def model_for(variant: tuple, positive_rate: float):
    name, family, first, second = variant
    weight = max((1 - positive_rate) / max(positive_rate, 1e-4), 1.0)
    if family == "rf":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=280, max_depth=first, min_samples_leaf=second,
                max_features="sqrt", class_weight="balanced_subsample", random_state=17, n_jobs=-1,
            )),
        ])
    if family == "extra":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=320, max_depth=first, min_samples_leaf=second,
                max_features="sqrt", class_weight="balanced", random_state=17, n_jobs=-1,
            )),
        ])
    if family == "hgb":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.05, max_leaf_nodes=first, l2_regularization=float(second),
                class_weight={0: 1.0, 1: weight}, random_state=17,
            )),
        ])
    if family == "lgb":
        return LGBMClassifier(
            objective="binary", n_estimators=320, learning_rate=0.03, num_leaves=first,
            max_depth=6, min_child_samples=second, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=3.0, reg_alpha=0.2, scale_pos_weight=weight, random_state=17,
            verbosity=-1, n_jobs=-1,
        )
    raise ValueError(name)


def evaluate_variant(frame: pd.DataFrame, target: pd.Series, dates: pd.Series, variant: tuple, start: int, end: int) -> pd.DataFrame:
    rows = []
    for year in range(start, end + 1):
        train = dates.dt.year < year
        test = dates.dt.year == year
        y_train = target.loc[train]
        y_test = target.loc[test]
        if len(y_train) < 500 or y_train.sum() < 20 or y_test.sum() < 2:
            continue
        model = model_for(variant, float(y_train.mean()))
        model.fit(frame.loc[train], y_train)
        metrics = event_metrics(y_test.to_numpy(), scores(model, frame.loc[test]), float(y_train.mean()))
        rows.append({"variant": variant[0], "year": year, **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    features, labels = build_features(raw)
    dates = pd.to_datetime(labels["date"])
    all_rows = []
    selected_rows = []

    for target_name in ("low", "high"):
        target = labels[target_name].astype(int)
        tuned = []
        for variant in VARIANTS:
            folds = evaluate_variant(features, target, dates, variant, FIRST_OOS_YEAR, TUNE_LAST_YEAR)
            if folds.empty:
                continue
            tuned.append(
                {
                    "target": target_name,
                    "variant": variant[0],
                    "tune_years": len(folds),
                    "tune_ap": folds["ap"].mean(),
                    "tune_precision": folds["top_precision"].mean(),
                    "tune_recall": folds["top_recall"].mean(),
                }
            )
            folds.insert(0, "target", target_name)
            folds["split"] = "tune"
            all_rows.append(folds)
        ranked = pd.DataFrame(tuned).sort_values(["tune_ap", "tune_precision"], ascending=False)
        choice = ranked.iloc[0]
        chosen_variant = next(item for item in VARIANTS if item[0] == choice["variant"])
        holdout = evaluate_variant(features, target, dates, chosen_variant, HOLDOUT_FIRST_YEAR, LAST_COMPLETE_OOS_YEAR)
        holdout.insert(0, "target", target_name)
        holdout["split"] = "holdout"
        all_rows.append(holdout)
        selected_rows.append(
            {
                **choice.to_dict(),
                "holdout_years": len(holdout),
                "holdout_ap": holdout["ap"].mean(),
                "holdout_precision": holdout["top_precision"].mean(),
                "holdout_recall": holdout["top_recall"].mean(),
            }
        )
        ranked.to_csv(OUT / f"tuning_{target_name}_ranking.csv", index=False, encoding="utf-8-sig")

    selected = pd.DataFrame(selected_rows)
    folds = pd.concat(all_rows, ignore_index=True)
    selected.to_csv(OUT / "tuning_selected.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(OUT / "tuning_fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tuning_period": f"{FIRST_OOS_YEAR}-{TUNE_LAST_YEAR}",
        "holdout_period": f"{HOLDOUT_FIRST_YEAR}-{LAST_COMPLETE_OOS_YEAR}",
        "selection_rule": "Choose by mean annual PR-AUC, then top-signal precision, using only the tuning period.",
        "selected": selected.to_dict(orient="records"),
    }
    (OUT / "tuning_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
