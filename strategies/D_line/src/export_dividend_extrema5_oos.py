from __future__ import annotations

"""Export leakage-safe 2020-2025 predictions of the selected 5-day extrema models."""

import json
from datetime import datetime

import numpy as np
import pandas as pd

from ablate_and_ensemble_dividend_extrema5 import groups
from fit_dividend_extrema5 import OUT, PROJECT, SYMBOL, build_features, scores
from tune_dividend_extrema5 import VARIANTS, model_for


HOLDOUT_START = 2020
HOLDOUT_END = 2025
LOW_MEMBERS = (
    ("rf_d10_l3", "all"),
    ("hgb_leaf15_l2_4", "price_only"),
    ("hgb_leaf15_l2_4", "all"),
)
HIGH_MEMBERS = (("lgb_leaf15_min20", "no_fourier"),)
VARIANT_BY_NAME = {item[0]: item for item in VARIANTS}


def percentile(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def top_mask(score: np.ndarray, rate: float) -> np.ndarray:
    count = max(1, int(round(len(score) * rate)))
    selected = np.zeros(len(score), dtype=bool)
    selected[np.argsort(score)[-count:]] = True
    return selected


def neighborhood_hit(label: np.ndarray, signal: np.ndarray, radius: int) -> np.ndarray:
    result = np.zeros(len(label), dtype=bool)
    for position in np.flatnonzero(signal):
        start = max(0, position - radius)
        end = min(len(label), position + radius + 1)
        result[position] = bool(label[start:end].any())
    return result


def predict_target(
    features: pd.DataFrame,
    target: pd.Series,
    dates: pd.Series,
    members: tuple[tuple[str, str], ...],
    feature_groups: dict[str, list[str]],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for year in range(HOLDOUT_START, HOLDOUT_END + 1):
        train = dates.dt.year < year
        test = dates.dt.year == year
        y_train = target.loc[train]
        if len(y_train) < 500 or y_train.sum() < 20 or not test.any():
            continue
        parts = []
        for variant_name, group_name in members:
            model = model_for(VARIANT_BY_NAME[variant_name], float(y_train.mean()))
            frame = features[feature_groups[group_name]]
            model.fit(frame.loc[train], y_train)
            parts.append(percentile(scores(model, frame.loc[test])))
        blend = np.mean(np.vstack(parts), axis=0)
        signal = top_mask(blend, float(y_train.mean()))
        actual = target.loc[test].to_numpy(dtype=int)
        frame = pd.DataFrame(
            {
                "date": dates.loc[test].to_numpy(),
                "year": year,
                "score": blend,
                "signal": signal.astype(int),
                "actual": actual,
            }
        )
        for radius in (0, 1, 2):
            frame[f"hit_within_{radius}d"] = neighborhood_hit(actual, signal, radius).astype(int)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def signal_summary(frame: pd.DataFrame, side: str) -> dict[str, object]:
    selected = frame.loc[frame["signal"] == 1]
    values: dict[str, object] = {
        "side": side,
        "predicted_points": int(len(selected)),
        "true_points": int(frame["actual"].sum()),
        "exact_precision": float(selected["hit_within_0d"].mean()),
    }
    for radius in (1, 2):
        values[f"within_{radius}d_precision"] = float(selected[f"hit_within_{radius}d"].mean())
    return values


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    features, labels = build_features(raw)
    dates = pd.to_datetime(labels["date"])
    feature_groups = groups(features)

    low = predict_target(features, labels["low"].astype(int), dates, LOW_MEMBERS, feature_groups)
    high = predict_target(features, labels["high"].astype(int), dates, HIGH_MEMBERS, feature_groups)
    panel = low.rename(
        columns={
            "score": "low_score",
            "signal": "low_signal",
            "actual": "actual_low",
            "hit_within_0d": "low_exact_hit",
            "hit_within_1d": "low_hit_within_1d",
            "hit_within_2d": "low_hit_within_2d",
        }
    ).merge(
        high.rename(
            columns={
                "score": "high_score",
                "signal": "high_signal",
                "actual": "actual_high",
                "hit_within_0d": "high_exact_hit",
                "hit_within_1d": "high_hit_within_1d",
                "hit_within_2d": "high_hit_within_2d",
            }
        ),
        on=["date", "year"],
        how="outer",
    )
    panel = panel.merge(raw[["date", "open", "high", "low", "close", "volume"]], on="date", how="left")
    panel = panel.sort_values("date")
    panel.to_csv(OUT / "holdout_predictions_2020_2025.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": SYMBOL,
        "holdout_period": f"{HOLDOUT_START}-{HOLDOUT_END}",
        "low_model": "rank-average(rf_d10_l3:all, hgb_leaf15_l2_4:price_only, hgb_leaf15_l2_4:all)",
        "high_model": "lgb_leaf15_min20:no_fourier",
        "low": signal_summary(low, "low"),
        "high": signal_summary(high, "high"),
        "note": "Exact means the predicted date is the labelled 5-day extremum. Within one/two days is reported separately and must not be treated as exact precision.",
    }
    (OUT / "holdout_prediction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
