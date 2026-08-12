from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v1_ml_extrema import ModelConfig, make_model, metric_at_best_threshold, tolerance_metrics


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "fit" / "extrema"
PANEL_FILE = BASE_DIR / "candidate" / "feature_panel.parquet"
OUT_DIR = BASE_DIR / "confirmation"

LOW_CONFIG = ModelConfig("low_confirm", 31, 40, 500, 0.035, 0.5)
HIGH_CONFIG = ModelConfig("high_confirm", 63, 24, 600, 0.03, 0.5)
CONFIRM_FEATURES = (
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ma_gap_5",
    "ma_gap_10",
    "ma_gap_20",
    "range_pos_10",
    "range_pos_20",
    "boll_z_10",
    "boll_z_20",
    "rsi_6",
    "rsi_14",
    "kdj_j_9",
    "macd_12_26",
    "volatility_10",
    "volatility_20",
    "volume_ratio_5_20",
    "candle_body",
    "close_in_bar",
    "market_mean_ret_5",
    "market_mean_ret_20",
    "market_breadth_ma20",
)


def base_features(panel: pd.DataFrame) -> list[str]:
    excluded = {
        "date",
        "symbol",
        "name",
        "source_file",
        "is_low",
        "is_high",
        "year",
        "bar_index",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "pct_change",
        "turnover_rate",
    }
    return [
        column
        for column in panel.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(panel[column])
    ]


def confirmation_dataset(panel: pd.DataFrame, target: str, delay: int) -> tuple[pd.DataFrame, list[str]]:
    candidate_column = "candidate_low_10" if target == "is_low" else "candidate_high_10"
    pieces = []
    for _, part in panel.groupby("symbol", sort=False):
        part = part.sort_values("date").copy()
        close = part["close"].astype(float)
        data = part[part[candidate_column].eq(1)].copy()
        candidate_indices = np.flatnonzero(part[candidate_column].eq(1).to_numpy())
        data["_row_id"] = data.index
        data["signal_date"] = [
            part.iloc[index + delay]["date"] if index + delay < len(part) else pd.NaT
            for index in candidate_indices
        ]
        data["label_known_date"] = [
            part.iloc[index + 10]["date"] if index + 10 < len(part) else pd.NaT
            for index in candidate_indices
        ]
        data["signal_year"] = pd.to_datetime(data["signal_date"]).dt.year
        if delay == 0:
            data["path_close_return"] = 0.0
            data["path_min_return"] = 0.0
            data["path_max_return"] = 0.0
            data["path_range"] = 0.0
            data["path_up_ratio"] = 0.0
        else:
            future_returns = pd.concat([close.shift(-step) / close - 1 for step in range(1, delay + 1)], axis=1)
            data["path_close_return"] = future_returns.iloc[:, -1].loc[data.index]
            data["path_min_return"] = future_returns.min(axis=1).loc[data.index]
            data["path_max_return"] = future_returns.max(axis=1).loc[data.index]
            data["path_range"] = data["path_max_return"] - data["path_min_return"]
            data["path_up_ratio"] = (future_returns > 0).mean(axis=1).loc[data.index]
        for feature in CONFIRM_FEATURES:
            if feature in part.columns:
                data[f"confirm_{feature}"] = part[feature].shift(-delay).loc[data.index]
        pieces.append(data)
    candidate = pd.concat(pieces, ignore_index=True)
    candidate = candidate[candidate["signal_date"].notna()].copy()
    added = [
        "path_close_return",
        "path_min_return",
        "path_max_return",
        "path_range",
        "path_up_ratio",
    ] + [f"confirm_{feature}" for feature in CONFIRM_FEATURES if f"confirm_{feature}" in candidate.columns]
    return candidate, base_features(panel) + added


def annual_walk(
    candidate: pd.DataFrame,
    features: list[str],
    target: str,
    config: ModelConfig,
    delay: int,
) -> np.ndarray:
    prediction = np.full(len(candidate), np.nan)
    y = candidate[target].to_numpy(dtype=np.int8)
    for year in sorted(candidate["signal_year"].dropna().astype(int).unique()):
        train_cutoff = pd.Timestamp(year=year, month=1, day=1)
        train_mask = candidate["label_known_date"].lt(train_cutoff).to_numpy()
        test_mask = candidate["signal_year"].eq(year).to_numpy()
        if train_mask.sum() < 300 or y[train_mask].sum() < 50:
            continue
        model = make_model(config, y[train_mask].mean(), 20260714 + year + delay * 100)
        model.fit(candidate.loc[train_mask, features], y[train_mask])
        prediction[test_mask] = model.predict_proba(candidate.loc[test_mask, features])[:, 1]
    return prediction


def expand_to_panel(
    panel: pd.DataFrame,
    candidate: pd.DataFrame,
    probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    full = np.full(len(panel), np.nan)
    signal_dates = np.full(len(panel), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    valid = np.isfinite(probability)
    years = candidate.loc[valid, "signal_year"].dropna().astype(int).unique()
    full[panel["year"].isin(years).to_numpy()] = 0.0
    row_ids = candidate.loc[valid, "_row_id"].to_numpy(dtype=int)
    full[row_ids] = probability[valid]
    signal_dates[row_ids] = pd.to_datetime(candidate.loc[valid, "signal_date"]).to_numpy()
    return full, signal_dates


def evaluate(panel: pd.DataFrame, target: str, probability: np.ndarray, delay: int) -> dict:
    valid = np.isfinite(probability)
    y = panel.loc[valid, target].to_numpy(dtype=np.int8)
    score = probability[valid]
    best = metric_at_best_threshold(y, score, beta=2.0)
    predicted = score >= best["threshold"]
    tolerance = tolerance_metrics(panel.loc[valid].copy(), target, predicted, tolerance=3)
    return {
        "target": target,
        "delay_days": delay,
        "samples": int(valid.sum()),
        "positives": int(y.sum()),
        **best,
        **tolerance,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PANEL_FILE)
    rows = []
    prediction_output = panel[["date", "symbol", "name", "close", "year", "is_low", "is_high"]].copy()
    for target, short, config in (
        ("is_low", "low", LOW_CONFIG),
        ("is_high", "high", HIGH_CONFIG),
    ):
        for delay in range(0, 6):
            print(f"[{target}] confirmation delay {delay}/5", flush=True)
            candidate, features = confirmation_dataset(panel, target, delay)
            candidate_probability = annual_walk(candidate, features, target, config, delay)
            full_probability, signal_dates = expand_to_panel(panel, candidate, candidate_probability)
            metrics = evaluate(panel, target, full_probability, delay)
            rows.append(metrics)
            prediction_output[f"{short}_delay_{delay}"] = full_probability
            prediction_output[f"{short}_signal_date_{delay}"] = signal_dates
    scan = pd.DataFrame(rows).sort_values(["target", "fbeta", "tolerance_f1"], ascending=[True, False, False])
    scan.to_csv(OUT_DIR / "delay_scan.csv", index=False, encoding="utf-8-sig")
    prediction_output.to_csv(OUT_DIR / "predictions.csv", index=False, encoding="utf-8-sig")
    best = scan.groupby("target", as_index=False).first()
    best.to_csv(OUT_DIR / "best_delays.csv", index=False, encoding="utf-8-sig")
    summary = {"model": "V1ML6_confirmation", "best": best.to_dict(orient="records")}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        scan[
            [
                "target",
                "delay_days",
                "precision",
                "recall",
                "fbeta",
                "tolerance_precision",
                "tolerance_recall",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
