from __future__ import annotations

"""Feature-group ablation and leakage-safe model blending for 510880 extrema labels."""

import json
from datetime import datetime

import numpy as np
import pandas as pd

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
from tune_dividend_extrema5 import HOLDOUT_FIRST_YEAR, TUNE_LAST_YEAR, VARIANTS, model_for


def groups(features: pd.DataFrame) -> dict[str, list[str]]:
    columns = list(features.columns)
    daily = [name for name in columns if name.startswith("D_")]
    non_daily = [name for name in columns if name.startswith(("W_", "B15_", "M_"))]
    fourier = [name for name in columns if name.startswith("fft")]
    volume = [name for name in columns if name.startswith("volume_")]
    price = [
        name
        for name in columns
        if name.startswith(("ret_", "ma_", "range_", "realized_", "drawdown_"))
        or name == "intraday_range"
    ]
    indicators = daily + non_daily
    return {
        "all": columns,
        "no_fourier": [name for name in columns if name not in fourier],
        "no_volume": [name for name in columns if name not in volume],
        "no_non_daily": [name for name in columns if name not in non_daily],
        "daily_price": daily + price,
        "daily_price_volume": daily + price + volume,
        "indicators": indicators,
        "daily_only": daily,
        "price_only": price + volume,
        "non_daily_only": non_daily,
    }


VARIANT_BY_NAME = {item[0]: item for item in VARIANTS}
TARGET_VARIANTS = {
    "low": ("rf_d10_l3", "hgb_leaf15_l2_4"),
    "high": ("lgb_leaf15_min20", "hgb_leaf15_l2_4"),
}


def yearly_predictions(
    features: pd.DataFrame,
    target: pd.Series,
    dates: pd.Series,
    columns: list[str],
    variant_name: str,
    start: int,
    end: int,
) -> tuple[pd.DataFrame, list[dict[str, np.ndarray]]]:
    rows: list[dict[str, object]] = []
    predictions: list[dict[str, np.ndarray]] = []
    variant = VARIANT_BY_NAME[variant_name]
    frame = features[columns]
    for year in range(start, end + 1):
        train = dates.dt.year < year
        test = dates.dt.year == year
        y_train = target.loc[train]
        y_test = target.loc[test]
        if len(y_train) < 500 or y_train.sum() < 20 or y_test.sum() < 2:
            continue
        model = model_for(variant, float(y_train.mean()))
        model.fit(frame.loc[train], y_train)
        score = scores(model, frame.loc[test])
        metrics = event_metrics(y_test.to_numpy(), score, float(y_train.mean()))
        rows.append({"year": year, **metrics})
        predictions.append({"year": year, "target": y_test.to_numpy(), "score": score, "rate": float(y_train.mean())})
    return pd.DataFrame(rows), predictions


def summarize(rows: pd.DataFrame) -> dict[str, float | int]:
    return {
        "years": int(len(rows)),
        "ap": float(rows["ap"].mean()),
        "top_precision": float(rows["top_precision"].mean()),
        "top_recall": float(rows["top_recall"].mean()),
        "signals": int(rows["signals"].sum()),
    }


def blend_predictions(prediction_sets: list[list[dict[str, np.ndarray]]], weights: list[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for items in zip(*prediction_sets):
        if len({int(item["year"]) for item in items}) != 1:
            raise ValueError("Ensemble folds are not aligned")
        rank_scores = []
        for item in items:
            score = pd.Series(item["score"]).rank(method="average", pct=True).to_numpy()
            rank_scores.append(score)
        mixed = np.average(np.vstack(rank_scores), axis=0, weights=weights)
        metrics = event_metrics(items[0]["target"], mixed, float(items[0]["rate"]))
        rows.append({"year": int(items[0]["year"]), **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    features, labels = build_features(raw)
    dates = pd.to_datetime(labels["date"])
    feature_groups = groups(features)
    grid_rows: list[dict[str, object]] = []
    cache: dict[tuple[str, str, str, int, int], list[dict[str, np.ndarray]]] = {}

    for target_name, variants in TARGET_VARIANTS.items():
        target = labels[target_name].astype(int)
        for variant in variants:
            for group_name, columns in feature_groups.items():
                tune_rows, tune_predictions = yearly_predictions(
                    features, target, dates, columns, variant, FIRST_OOS_YEAR, TUNE_LAST_YEAR
                )
                cache[(target_name, variant, group_name, FIRST_OOS_YEAR, TUNE_LAST_YEAR)] = tune_predictions
                result = {
                    "target": target_name,
                    "kind": "single",
                    "name": f"{variant}__{group_name}",
                    "members": f"{variant}:{group_name}",
                    "features": len(columns),
                    **summarize(tune_rows),
                }
                grid_rows.append(result)

    grid = pd.DataFrame(grid_rows)
    selected_rows: list[dict[str, object]] = []
    holdout_rows: list[pd.DataFrame] = []

    for target_name in TARGET_VARIANTS:
        ranked = grid.loc[grid["target"] == target_name].sort_values(
            ["ap", "top_precision", "top_recall"], ascending=False
        ).reset_index(drop=True)
        ranked.to_csv(OUT / f"ablation_{target_name}_ranking.csv", index=False, encoding="utf-8-sig")
        top = ranked.head(3)
        ensemble_specs = []
        if len(top) >= 2:
            ensemble_specs.append(("blend_top2_equal", [0, 1], [0.5, 0.5]))
        if len(top) >= 3:
            ensemble_specs.extend(
                [
                    ("blend_top3_equal", [0, 1, 2], [1 / 3, 1 / 3, 1 / 3]),
                    ("blend_top3_weighted", [0, 1, 2], [0.55, 0.30, 0.15]),
                ]
            )

        ensemble_rows: list[dict[str, object]] = []
        for name, indexes, weights in ensemble_specs:
            sets = []
            members = []
            for index in indexes:
                record = top.iloc[index]
                variant, group_name = str(record["members"]).split(":", 1)
                sets.append(cache[(target_name, variant, group_name, FIRST_OOS_YEAR, TUNE_LAST_YEAR)])
                members.append(str(record["members"]))
            metrics = blend_predictions(sets, weights)
            ensemble_rows.append(
                {
                    "target": target_name,
                    "kind": "blend",
                    "name": name,
                    "members": " | ".join(members),
                    "features": int(top.iloc[indexes]["features"].max()),
                    **summarize(metrics),
                }
            )

        all_candidates = pd.concat([ranked, pd.DataFrame(ensemble_rows)], ignore_index=True).sort_values(
            ["ap", "top_precision", "top_recall"], ascending=False
        ).reset_index(drop=True)
        all_candidates.to_csv(OUT / f"ablation_blend_{target_name}_ranking.csv", index=False, encoding="utf-8-sig")
        choice = all_candidates.iloc[0].to_dict()
        choice["tune_rank"] = 1
        selected_rows.append(choice)

        # Validate exactly the tuning-selected option against untouched future years.
        target = labels[target_name].astype(int)
        if choice["kind"] == "single":
            variant, group_name = str(choice["members"]).split(":", 1)
            holdout, _ = yearly_predictions(
                features, target, dates, feature_groups[group_name], variant, HOLDOUT_FIRST_YEAR, 2025
            )
        else:
            members = str(choice["members"]).split(" | ")
            prediction_sets = []
            for member in members:
                variant, group_name = member.split(":", 1)
                _, predicted = yearly_predictions(
                    features, target, dates, feature_groups[group_name], variant, HOLDOUT_FIRST_YEAR, 2025
                )
                prediction_sets.append(predicted)
            weights = [1 / len(prediction_sets)] * len(prediction_sets)
            if choice["name"] == "blend_top3_weighted":
                weights = [0.55, 0.30, 0.15]
            holdout = blend_predictions(prediction_sets, weights)
        metrics = summarize(holdout)
        holdout.insert(0, "target", target_name)
        holdout.insert(1, "selected_name", str(choice["name"]))
        holdout_rows.append(holdout)
        selected_rows[-1].update({f"holdout_{key}": value for key, value in metrics.items()})

    selected = pd.DataFrame(selected_rows)
    holdout_frame = pd.concat(holdout_rows, ignore_index=True)
    selected.to_csv(OUT / "ablation_selected.csv", index=False, encoding="utf-8-sig")
    holdout_frame.to_csv(OUT / "ablation_holdout_yearly.csv", index=False, encoding="utf-8-sig")
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": SYMBOL,
        "label": "Independent 5-day extrema with 0.5% / five-day near-equal cluster merging.",
        "tuning_period": f"{FIRST_OOS_YEAR}-{TUNE_LAST_YEAR}",
        "untouched_holdout": "2020-2025",
        "selection_rule": "Mean annual PR-AUC, then top-signal precision and recall. Blends use only tuning-ranked base models.",
        "selected": selected.to_dict(orient="records"),
    }
    (OUT / "ablation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
