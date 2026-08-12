from __future__ import annotations

"""Create a current-year prediction panel using models frozen after 2025 selection."""

import json
from datetime import datetime

import numpy as np
import pandas as pd

from ablate_and_ensemble_dividend_extrema5 import groups
from fit_dividend_extrema5 import OUT, PROJECT, SYMBOL, build_features, scores
from tune_dividend_extrema5 import VARIANTS, model_for


LOW_MEMBERS = (
    ("rf_d10_l3", "all"),
    ("hgb_leaf15_l2_4", "price_only"),
    ("hgb_leaf15_l2_4", "all"),
)
HIGH_MEMBERS = (("lgb_leaf15_min20", "no_fourier"),)
VARIANTS_BY_NAME = {item[0]: item for item in VARIANTS}
TARGET_YEAR = 2026


def percentile(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def top_mask(score: np.ndarray, rate: float) -> np.ndarray:
    count = max(1, int(round(len(score) * rate)))
    result = np.zeros(len(score), dtype=int)
    result[np.argsort(score)[-count:]] = 1
    return result


def predict(features: pd.DataFrame, target: pd.Series, dates: pd.Series, members, sets) -> tuple[np.ndarray, np.ndarray]:
    train = dates.dt.year < TARGET_YEAR
    test = dates.dt.year == TARGET_YEAR
    train_target = target.loc[train]
    parts = []
    for variant_name, group_name in members:
        model = model_for(VARIANTS_BY_NAME[variant_name], float(train_target.mean()))
        frame = features[sets[group_name]]
        model.fit(frame.loc[train], train_target)
        parts.append(percentile(scores(model, frame.loc[test])))
    score = np.mean(np.vstack(parts), axis=0)
    return score, top_mask(score, float(train_target.mean()))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    features, labels = build_features(raw, include_unlabeled_tail=True)
    dates = pd.to_datetime(labels["date"])
    sets = groups(features)
    low_score, low_signal = predict(features, labels["low"].astype(int), dates, LOW_MEMBERS, sets)
    high_score, high_signal = predict(features, labels["high"].astype(int), dates, HIGH_MEMBERS, sets)
    panel = labels.loc[dates.dt.year == TARGET_YEAR, ["date", "low", "high"]].rename(
        columns={"low": "actual_low", "high": "actual_high"}
    )
    panel["low_score"] = low_score
    panel["high_score"] = high_score
    panel["low_signal"] = low_signal
    panel["high_signal"] = high_signal
    panel = panel.merge(raw[["date", "open", "high", "low", "close", "volume"]], on="date", how="left")
    panel.to_csv(OUT / "live_predictions_2026.csv", index=False, encoding="utf-8-sig")
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": SYMBOL,
        "data_latest_date": str(panel["date"].max().date()),
        "rows": int(len(panel)),
        "low_candidates": int(panel["low_signal"].sum()),
        "high_candidates": int(panel["high_signal"].sum()),
        "latest_low_score": float(panel.iloc[-1]["low_score"]),
        "latest_high_score": float(panel.iloc[-1]["high_score"]),
        "warning": "The last five dates do not have completed future windows, so actual high/low labels there are intentionally unavailable.",
    }
    (OUT / "live_prediction_summary_2026.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
