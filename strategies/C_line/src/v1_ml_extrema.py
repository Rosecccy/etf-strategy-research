from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

_MPL_CACHE = Path(__file__).resolve().parents[1] / "out" / "matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw" / "etf"
OUT_DIR = ROOT / "fit" / "extrema"
MODEL_DIR = OUT_DIR / "models"
LOCKED_LABEL_PANEL = ROOT / "fit" / "extrema" / "extrema_feature_panel.csv"
LABEL_WINDOW = 10
RANDOM_SEEDS = (20260713, 510050, 159915)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_leaves: int
    min_child_samples: int
    n_estimators: int
    learning_rate: float
    class_weight_multiplier: float
    max_depth: int = -1
    reg_alpha: float = 0.2
    reg_lambda: float = 1.0
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85


CONFIGS = (
    ModelConfig("L31_M40_W05", 31, 40, 450, 0.04, 0.5),
    ModelConfig("L31_M40_W10", 31, 40, 450, 0.04, 1.0),
    ModelConfig("L63_M24_W05", 63, 24, 600, 0.035, 0.5),
    ModelConfig("L63_M24_W10", 63, 24, 600, 0.035, 1.0),
    ModelConfig("L127_M12_W05", 127, 12, 800, 0.025, 0.5),
    ModelConfig("L127_M12_W10", 127, 12, 800, 0.025, 1.0),
    ModelConfig("L255_M6_W05", 255, 6, 1000, 0.02, 0.5),
    ModelConfig("L255_M6_W10", 255, 6, 1000, 0.02, 1.0),
)


def safe_div(left: pd.Series, right: pd.Series) -> pd.Series:
    return left / right.replace(0, np.nan)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=max(2, span // 2)).mean()


def rsi(series: pd.Series, window: int) -> pd.Series:
    diff = series.diff()
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    avg_up = up.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_down = down.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = safe_div(avg_up, avg_down)
    return 100 - 100 / (1 + rs)


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x -= x.mean()
    denom = float(np.square(x).sum())

    def slope(values: np.ndarray) -> float:
        mean = float(np.mean(values))
        if not np.isfinite(mean) or mean == 0:
            return np.nan
        return float(np.dot(values - mean, x) / denom / mean)

    return series.rolling(window, min_periods=window).apply(slope, raw=True)


def bars_since_extreme(series: pd.Series, window: int, mode: str) -> pd.Series:
    if mode == "low":
        fn = lambda x: len(x) - 1 - int(np.argmin(x))
    else:
        fn = lambda x: len(x) - 1 - int(np.argmax(x))
    return series.rolling(window, min_periods=window).apply(fn, raw=True)


def original_extrema_labels(close: pd.Series, window: int = LABEL_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    values = close.to_numpy(dtype=float)
    lows: list[int] = []
    highs: list[int] = []
    for i in range(window, len(values) - window):
        before = values[i - window : i]
        after = values[i + 1 : i + window + 1]
        if np.all(before > values[i]) and np.all(after > values[i]):
            lows.append(i)
        if np.all(before < values[i]) and np.all(after < values[i]):
            highs.append(i)

    events = sorted([(i, "low") for i in lows] + [(i, "high") for i in highs])
    paired_lows: set[int] = set()
    paired_highs: set[int] = set()
    active_low: int | None = None
    best_high: int | None = None
    for index, kind in events:
        if kind == "low":
            if active_low is not None and best_high is not None:
                paired_lows.add(active_low)
                paired_highs.add(best_high)
            active_low = index
            best_high = None
        elif active_low is not None and index > active_low:
            if best_high is None or values[index] > values[best_high]:
                best_high = index
    if active_low is not None and best_high is not None:
        paired_lows.add(active_low)
        paired_highs.add(best_high)

    low_label = np.zeros(len(close), dtype=np.int8)
    high_label = np.zeros(len(close), dtype=np.int8)
    low_label[list(paired_lows)] = 1
    high_label[list(paired_highs)] = 1
    return low_label, high_label


def add_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float).replace(0, np.nan)
    amount = df.get("amount", pd.Series(np.nan, index=df.index)).astype(float).replace(0, np.nan)
    returns = close.pct_change()

    df["candle_body"] = safe_div(close - open_, open_)
    df["candle_range"] = safe_div(high - low, close.shift(1))
    df["close_in_bar"] = safe_div(close - low, high - low)
    df["upper_shadow"] = safe_div(high - np.maximum(open_, close), close.shift(1))
    df["lower_shadow"] = safe_div(np.minimum(open_, close) - low, close.shift(1))
    df["open_gap"] = safe_div(open_, close.shift(1)) - 1
    df["true_range"] = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
    ).max(axis=1) / close.shift(1)

    for lag in (1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 90, 120):
        df[f"ret_{lag}"] = close.pct_change(lag)

    for window in (3, 5, 8, 10, 15, 20, 30, 40, 60, 90, 120):
        ma = close.rolling(window, min_periods=window).mean()
        ema_value = ema(close, window)
        roll_low = low.rolling(window, min_periods=window).min()
        roll_high = high.rolling(window, min_periods=window).max()
        std = returns.rolling(window, min_periods=window).std()
        downside = returns.where(returns < 0, 0).rolling(window, min_periods=window).std()
        df[f"ma_gap_{window}"] = safe_div(close, ma) - 1
        df[f"ema_gap_{window}"] = safe_div(close, ema_value) - 1
        df[f"range_pos_{window}"] = safe_div(close - roll_low, roll_high - roll_low)
        df[f"dd_high_{window}"] = safe_div(close, roll_high) - 1
        df[f"up_low_{window}"] = safe_div(close, roll_low) - 1
        df[f"volatility_{window}"] = std
        df[f"downside_vol_{window}"] = downside
        df[f"return_sharpe_{window}"] = safe_div(returns.rolling(window, min_periods=window).mean(), std)
        if window in (5, 10, 20, 40, 60, 120):
            df[f"slope_{window}"] = rolling_slope(close, window)
        if window in (10, 20, 60, 120):
            df[f"bars_from_low_{window}"] = bars_since_extreme(close, window, "low")
            df[f"bars_from_high_{window}"] = bars_since_extreme(close, window, "high")

    for window in (5, 10, 20, 40, 60):
        mean = close.rolling(window, min_periods=window).mean()
        std = close.rolling(window, min_periods=window).std()
        df[f"boll_z_{window}"] = safe_div(close - mean, std)
        df[f"boll_width_{window}"] = safe_div(4 * std, mean)

    for window in (3, 5, 6, 9, 14, 21, 28):
        df[f"rsi_{window}"] = rsi(close, window)

    for window in (5, 9, 14, 21, 34):
        roll_low = low.rolling(window, min_periods=window).min()
        roll_high = high.rolling(window, min_periods=window).max()
        rsv = safe_div(close - roll_low, roll_high - roll_low) * 100
        k = rsv.ewm(alpha=1 / 3, adjust=False, min_periods=3).mean()
        d = k.ewm(alpha=1 / 3, adjust=False, min_periods=3).mean()
        df[f"kdj_k_{window}"] = k
        df[f"kdj_d_{window}"] = d
        df[f"kdj_j_{window}"] = 3 * k - 2 * d

    for fast, slow, signal in ((6, 13, 5), (12, 26, 9), (24, 52, 18)):
        dif = ema(close, fast) - ema(close, slow)
        dea = ema(dif, signal)
        scale = close.rolling(slow, min_periods=slow).mean()
        df[f"dif_{fast}_{slow}"] = safe_div(dif, scale)
        df[f"macd_{fast}_{slow}"] = safe_div(dif - dea, scale)
        df[f"macd_slope_{fast}_{slow}"] = safe_div(dif.diff(3), scale)

    for short, long in ((3, 10), (5, 20), (10, 40), (20, 60), (20, 120)):
        short_volume = volume.rolling(short, min_periods=short).mean()
        long_volume = volume.rolling(long, min_periods=long).mean()
        short_amount = amount.rolling(short, min_periods=short).mean()
        long_amount = amount.rolling(long, min_periods=long).mean()
        df[f"volume_ratio_{short}_{long}"] = safe_div(short_volume, long_volume)
        df[f"amount_ratio_{short}_{long}"] = safe_div(short_amount, long_amount)

    signed_volume = np.sign(returns.fillna(0)) * np.log1p(volume)
    for window in (5, 10, 20, 60):
        df[f"signed_volume_{window}"] = signed_volume.rolling(window, min_periods=window).mean()
        df[f"price_volume_corr_{window}"] = returns.rolling(window, min_periods=window).corr(
            np.log1p(volume).diff()
        )

    df["ret_accel_5_20"] = df["ret_5"] - df["ret_20"] / 4
    df["ret_accel_10_60"] = df["ret_10"] - df["ret_60"] / 6
    df["ma_cross_5_20"] = safe_div(close.rolling(5).mean(), close.rolling(20).mean()) - 1
    df["ma_cross_20_60"] = safe_div(close.rolling(20).mean(), close.rolling(60).mean()) - 1
    df["ma_cross_60_120"] = safe_div(close.rolling(60).mean(), close.rolling(120).mean()) - 1
    df["up_day_ratio_10"] = (returns > 0).rolling(10, min_periods=10).mean()
    df["up_day_ratio_20"] = (returns > 0).rolling(20, min_periods=20).mean()
    trailing_low_10 = close.rolling(10, min_periods=10).min()
    trailing_high_10 = close.rolling(10, min_periods=10).max()
    new_low_10 = close.eq(trailing_low_10).astype(float)
    new_high_10 = close.eq(trailing_high_10).astype(float)
    previous_low_10 = close.shift(1).rolling(10, min_periods=10).min()
    previous_high_10 = close.shift(1).rolling(10, min_periods=10).max()
    df["candidate_low_10"] = new_low_10
    df["candidate_high_10"] = new_high_10
    df["low_break_strength_10"] = safe_div(close, previous_low_10) - 1
    df["high_break_strength_10"] = safe_div(close, previous_high_10) - 1
    for window in (5, 10, 20, 40):
        df[f"new_low_count_{window}"] = new_low_10.rolling(window, min_periods=window).sum()
        df[f"new_high_count_{window}"] = new_high_10.rolling(window, min_periods=window).sum()
    df["age_days"] = np.arange(len(df), dtype=float)
    return df


def build_panel() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        raw = pd.read_csv(path)
        required = {"date", "symbol", "open", "high", "low", "close", "volume"}
        if not required.issubset(raw.columns):
            continue
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        raw["symbol"] = raw["symbol"].astype(str).str.zfill(6)
        raw["name"] = raw.get("name", "").fillna("")
        low_label, high_label = original_extrema_labels(raw["close"])
        enriched = add_features(raw)
        enriched["is_low"] = low_label
        enriched["is_high"] = high_label
        enriched["year"] = enriched["date"].dt.year
        enriched["bar_index"] = np.arange(len(enriched), dtype=int)
        parts.append(enriched)
    if not parts:
        raise RuntimeError(f"No ETF CSV files found in {RAW_DIR}")
    panel = pd.concat(parts, ignore_index=True, sort=False)

    # The research target is the previously approved V1 label set. Raw files now
    # contain a longer warm-up history, so lock both the sample dates and labels
    # to that frozen panel instead of silently creating a different target.
    locked = pd.read_csv(
        LOCKED_LABEL_PANEL,
        usecols=["date", "symbol", "is_paired_low", "is_paired_high"],
        dtype={"symbol": str},
    )
    locked["date"] = pd.to_datetime(locked["date"])
    locked["symbol"] = locked["symbol"].str.zfill(6)
    panel = panel.drop(columns=["is_low", "is_high"])
    panel = panel.merge(locked, on=["date", "symbol"], how="inner", validate="one_to_one")
    panel = panel.rename(columns={"is_paired_low": "is_low", "is_paired_high": "is_high"})

    market_inputs = ["ret_1", "ret_5", "ret_20", "ret_60", "ma_gap_20", "ma_gap_60", "range_pos_20"]
    for feature in market_inputs:
        panel[f"market_mean_{feature}"] = panel.groupby("date")[feature].transform("mean")
        panel[f"market_rank_{feature}"] = panel.groupby("date")[feature].rank(pct=True)
    panel["market_breadth_ma20"] = panel.groupby("date")["ma_gap_20"].transform(lambda x: (x > 0).mean())
    panel["market_breadth_ma60"] = panel.groupby("date")["ma_gap_60"].transform(lambda x: (x > 0).mean())
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def feature_columns(panel: pd.DataFrame) -> list[str]:
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
    return [c for c in panel.columns if c not in excluded and pd.api.types.is_numeric_dtype(panel[c])]


def make_model(config: ModelConfig, positive_rate: float, seed: int) -> lgb.LGBMClassifier:
    positive_rate = max(min(float(positive_rate), 0.49), 1e-5)
    scale_pos_weight = (1 - positive_rate) / positive_rate * config.class_weight_multiplier
    return lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        max_depth=config.max_depth,
        min_child_samples=config.min_child_samples,
        subsample=config.bagging_fraction,
        subsample_freq=1,
        colsample_bytree=config.feature_fraction,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def metric_at_best_threshold(y_true: np.ndarray, probability: np.ndarray, beta: float = 2.0) -> dict:
    prevalence = max(int(y_true.sum()), 1)
    thresholds = np.unique(
        np.r_[np.linspace(0.02, 0.98, 97), np.quantile(probability, np.linspace(0.70, 0.999, 100))]
    )
    best: dict | None = None
    for threshold in thresholds:
        predicted = probability >= threshold
        signal_ratio = int(predicted.sum()) / prevalence
        if predicted.sum() == 0 or signal_ratio > 3.0:
            continue
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, predicted, average="binary", zero_division=0
        )
        f_beta = fbeta_score(y_true, predicted, beta=beta, zero_division=0)
        mcc = matthews_corrcoef(y_true, predicted)
        row = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "fbeta": float(f_beta),
            "mcc": float(mcc),
            "signals": int(predicted.sum()),
            "signal_ratio": float(signal_ratio),
        }
        if best is None or (row["fbeta"], row["mcc"]) > (best["fbeta"], best["mcc"]):
            best = row
    return best or {
        "threshold": 1.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "fbeta": 0.0,
        "mcc": 0.0,
        "signals": 0,
        "signal_ratio": 0.0,
    }


def rolling_splits(panel: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    boundaries = ((2017, 2018, 2019), (2019, 2020, 2021), (2021, 2022, 2023), (2023, 2024, 2026))
    rows = []
    for train_end, test_start, test_end in boundaries:
        train_mask = panel["year"].le(train_end).to_numpy()
        test_mask = panel["year"].between(test_start, test_end).to_numpy()
        rows.append((f"{test_start}-{test_end}", train_mask, test_mask))
    return rows


def model_search(panel: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    y = panel[target].to_numpy(dtype=np.int8)
    rows: list[dict] = []
    for config_index, config in enumerate(CONFIGS, 1):
        fold_rows = []
        print(f"[{target}] config {config_index}/{len(CONFIGS)}: {config.name}", flush=True)
        for fold_name, train_mask, test_mask in rolling_splits(panel):
            model = make_model(config, y[train_mask].mean(), RANDOM_SEEDS[0])
            model.fit(panel.loc[train_mask, features], y[train_mask])
            probability = model.predict_proba(panel.loc[test_mask, features])[:, 1]
            y_test = y[test_mask]
            threshold_metrics = metric_at_best_threshold(y_test, probability, beta=2.0)
            fold_rows.append(
                {
                    "fold": fold_name,
                    "average_precision": average_precision_score(y_test, probability),
                    "roc_auc": roc_auc_score(y_test, probability),
                    **threshold_metrics,
                }
            )
        fold_df = pd.DataFrame(fold_rows)
        row = {
            "target": target,
            **asdict(config),
            "mean_average_precision": float(fold_df["average_precision"].mean()),
            "std_average_precision": float(fold_df["average_precision"].std(ddof=0)),
            "mean_roc_auc": float(fold_df["roc_auc"].mean()),
            "mean_f2": float(fold_df["fbeta"].mean()),
            "mean_precision": float(fold_df["precision"].mean()),
            "mean_recall": float(fold_df["recall"].mean()),
        }
        row["selection_score"] = (
            row["mean_average_precision"]
            + 0.25 * row["mean_f2"]
            - 0.10 * row["std_average_precision"]
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("selection_score", ascending=False).reset_index(drop=True)


def config_from_row(row: pd.Series) -> ModelConfig:
    return ModelConfig(
        name=str(row["name"]),
        num_leaves=int(row["num_leaves"]),
        min_child_samples=int(row["min_child_samples"]),
        n_estimators=int(row["n_estimators"]),
        learning_rate=float(row["learning_rate"]),
        class_weight_multiplier=float(row["class_weight_multiplier"]),
        max_depth=int(row["max_depth"]),
        reg_alpha=float(row["reg_alpha"]),
        reg_lambda=float(row["reg_lambda"]),
        feature_fraction=float(row["feature_fraction"]),
        bagging_fraction=float(row["bagging_fraction"]),
    )


def fit_full_ensemble(
    panel: pd.DataFrame, features: list[str], target: str, config: ModelConfig
) -> tuple[np.ndarray, pd.DataFrame]:
    y = panel[target].to_numpy(dtype=np.int8)
    probabilities = []
    importances = []
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for seed in RANDOM_SEEDS:
        model = make_model(config, y.mean(), seed)
        model.fit(panel[features], y)
        probabilities.append(model.predict_proba(panel[features])[:, 1])
        model.booster_.save_model(str(MODEL_DIR / f"{target}_{seed}.txt"))
        importances.append(model.booster_.feature_importance(importance_type="gain"))
    importance = pd.DataFrame(
        {
            "target": target,
            "feature": features,
            "gain": np.mean(np.vstack(importances), axis=0),
        }
    ).sort_values("gain", ascending=False)
    importance["gain_share"] = importance["gain"] / importance["gain"].sum()
    return np.mean(np.vstack(probabilities), axis=0), importance


def annual_walk_forward(
    panel: pd.DataFrame, features: list[str], target: str, config: ModelConfig
) -> np.ndarray:
    prediction = np.full(len(panel), np.nan, dtype=float)
    years = sorted(panel["year"].unique())
    y = panel[target].to_numpy(dtype=np.int8)
    for year in years:
        test_mask = panel["year"].eq(year).to_numpy()
        train_mask = panel["year"].lt(year).to_numpy()
        if train_mask.sum() < 1000 or y[train_mask].sum() < 50:
            continue
        print(f"[{target}] walk-forward year {year}", flush=True)
        model = make_model(config, y[train_mask].mean(), RANDOM_SEEDS[0] + int(year))
        model.fit(panel.loc[train_mask, features], y[train_mask])
        prediction[test_mask] = model.predict_proba(panel.loc[test_mask, features])[:, 1]
    return prediction


def etf_group_oof(panel: pd.DataFrame, features: list[str], target: str, config: ModelConfig) -> np.ndarray:
    prediction = np.full(len(panel), np.nan, dtype=float)
    y = panel[target].to_numpy(dtype=np.int8)
    groups = panel["symbol"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    for fold, (train_index, test_index) in enumerate(splitter.split(panel, y, groups), 1):
        print(f"[{target}] ETF-isolation fold {fold}/5", flush=True)
        model = make_model(config, y[train_index].mean(), RANDOM_SEEDS[0] + fold)
        model.fit(panel.iloc[train_index][features], y[train_index])
        prediction[test_index] = model.predict_proba(panel.iloc[test_index][features])[:, 1]
    return prediction


def tolerance_metrics(panel: pd.DataFrame, target: str, predicted: np.ndarray, tolerance: int = 3) -> dict:
    true_total = 0
    pred_total = 0
    true_matched = 0
    pred_matched = 0
    for _, part in panel.assign(_pred=predicted).groupby("symbol", sort=False):
        truth = part[target].to_numpy(dtype=bool)
        pred = part["_pred"].to_numpy(dtype=bool)
        truth_positions = np.flatnonzero(truth)
        pred_positions = np.flatnonzero(pred)
        true_total += len(truth_positions)
        pred_total += len(pred_positions)
        if len(pred_positions):
            pred_matched += sum(np.any(np.abs(truth_positions - p) <= tolerance) for p in pred_positions)
        if len(truth_positions):
            true_matched += sum(np.any(np.abs(pred_positions - p) <= tolerance) for p in truth_positions)
    precision = pred_matched / pred_total if pred_total else 0.0
    recall = true_matched / true_total if true_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tolerance_days": tolerance,
        "tolerance_precision": precision,
        "tolerance_recall": recall,
        "tolerance_f1": f1,
    }


def evaluate_predictions(
    panel: pd.DataFrame, target: str, probability: np.ndarray, evaluation: str
) -> tuple[dict, pd.DataFrame]:
    valid = np.isfinite(probability)
    y = panel.loc[valid, target].to_numpy(dtype=np.int8)
    p = probability[valid]
    best_f2 = metric_at_best_threshold(y, p, beta=2.0)
    best_f1 = metric_at_best_threshold(y, p, beta=1.0)
    selected = p >= best_f2["threshold"]
    tolerance = tolerance_metrics(panel.loc[valid].copy(), target, selected, tolerance=3)
    summary = {
        "target": target,
        "evaluation": evaluation,
        "samples": int(valid.sum()),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        **{f"f2_{key}": value for key, value in best_f2.items()},
        **{f"f1_{key}": value for key, value in best_f1.items()},
        **tolerance,
    }
    signals = panel.loc[valid, ["date", "symbol", "name", "close", target]].copy()
    signals["target"] = target
    signals["evaluation"] = evaluation
    signals["probability"] = p
    signals["is_signal"] = selected
    signals = signals[signals["is_signal"]].drop(columns="is_signal")
    return summary, signals


def combined_turning_metrics(predictions: pd.DataFrame, threshold_rows: pd.DataFrame) -> dict:
    thresholds = {
        (row["target"], row["evaluation"]): float(row["f2_threshold"])
        for _, row in threshold_rows.iterrows()
    }
    output = {}
    for evaluation, low_col, high_col in (
        ("full_fit", "low_full", "high_full"),
        ("walk_forward", "low_walk", "high_walk"),
        ("etf_oof", "low_etf_oof", "high_etf_oof"),
    ):
        low_threshold = thresholds.get(("is_low", evaluation))
        high_threshold = thresholds.get(("is_high", evaluation))
        if low_threshold is None or high_threshold is None:
            continue
        valid = predictions[[low_col, high_col]].notna().all(axis=1)
        low_score = predictions.loc[valid, low_col]
        high_score = predictions.loc[valid, high_col]
        low_signal = (low_score >= low_threshold) & (low_score > high_score)
        high_signal = (high_score >= high_threshold) & (high_score > low_score)
        output[evaluation] = {
            "low_signals_after_mutual_exclusion": int(low_signal.sum()),
            "high_signals_after_mutual_exclusion": int(high_signal.sum()),
            "ambiguous_signals_removed": int(
                ((low_score >= low_threshold) & (high_score >= high_threshold)).sum()
            ),
        }
    return output


def write_report(summary: dict, evaluations: pd.DataFrame, importance: pd.DataFrame) -> None:
    lines = [
        "# V1ML 高低点拟合报告",
        "",
        "V1ML 的唯一学习目标，是拟合已经确认的 V1 原始高低点。标签仍是前后各 10 个交易日的局部极值，并按低点到后续最高高点配对。指标只使用当日收盘时已经知道的数据。",
        "",
        "## 三种验证口径",
        "",
        "- 全样本拟合：衡量模型记住既有高低点的理论上限，不代表实盘能力。",
        "- 年度滚动：每年只用以前年度训练，再预测下一年，是主要可落地口径。",
        "- ETF 隔离：整只 ETF 不参加训练，检验形态能否迁移到未见品种。",
        "",
        "## 结果",
        "",
        "| 目标 | 口径 | PR-AUC | ROC-AUC | 精确率 | 召回率 | F2 | ±3日召回 | 信号/真值 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    names = {"is_low": "低点", "is_high": "高点"}
    eval_names = {"full_fit": "全样本拟合", "walk_forward": "年度滚动", "etf_oof": "ETF隔离"}
    for _, row in evaluations.iterrows():
        lines.append(
            f"| {names[row['target']]} | {eval_names[row['evaluation']]} | {row['average_precision']:.3f} | "
            f"{row['roc_auc']:.3f} | {row['f2_precision']:.2%} | {row['f2_recall']:.2%} | "
            f"{row['f2_fbeta']:.3f} | {row['tolerance_recall']:.2%} | {row['f2_signal_ratio']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 重要说明",
            "",
            "V1 标签需要未来 10 个交易日才能最终确认，所以全样本拟合只能作为形态教师。年度滚动结果才反映当日可计算特征对未来转折的识别能力。F2 比普通 F1 更重视召回，但信号数量被限制为真实高低点数量的 3 倍以内，避免每天都报信号。",
            "",
            "## 主要特征",
            "",
            "| 目标 | 排名 | 特征 | 增益占比 |",
            "| --- | ---: | --- | ---: |",
        ]
    )
    for target in ("is_low", "is_high"):
        top = importance[importance["target"] == target].head(15)
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            lines.append(f"| {names[target]} | {rank} | {row['feature']} | {row['gain_share']:.2%} |")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            summary["conclusion"],
        ]
    )
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Building enriched feature panel...", flush=True)
    panel = build_panel()
    features = feature_columns(panel)
    print(
        f"Panel: {len(panel):,} rows, {panel['symbol'].nunique()} ETFs, {len(features)} features, "
        f"low={panel['is_low'].sum()}, high={panel['is_high'].sum()}",
        flush=True,
    )

    search_parts = []
    best_configs: dict[str, ModelConfig] = {}
    for target in ("is_low", "is_high"):
        search = model_search(panel, features, target)
        search_parts.append(search)
        best_configs[target] = config_from_row(search.iloc[0])
        print(f"Best {target}: {best_configs[target].name}", flush=True)
    search_results = pd.concat(search_parts, ignore_index=True)
    search_results.to_csv(OUT_DIR / "model_search.csv", index=False, encoding="utf-8-sig")

    predictions = panel[["date", "symbol", "name", "close", "year", "bar_index", "is_low", "is_high"]].copy()
    importance_parts = []
    evaluation_rows = []
    signal_parts = []
    for target, short in (("is_low", "low"), ("is_high", "high")):
        config = best_configs[target]
        full_probability, importance = fit_full_ensemble(panel, features, target, config)
        walk_probability = annual_walk_forward(panel, features, target, config)
        etf_probability = etf_group_oof(panel, features, target, config)
        predictions[f"{short}_full"] = full_probability
        predictions[f"{short}_walk"] = walk_probability
        predictions[f"{short}_etf_oof"] = etf_probability
        importance_parts.append(importance)
        for evaluation, probability in (
            ("full_fit", full_probability),
            ("walk_forward", walk_probability),
            ("etf_oof", etf_probability),
        ):
            metrics, signals = evaluate_predictions(panel, target, probability, evaluation)
            evaluation_rows.append(metrics)
            signal_parts.append(signals)

    evaluations = pd.DataFrame(evaluation_rows)
    importance = pd.concat(importance_parts, ignore_index=True)
    signals = pd.concat(signal_parts, ignore_index=True)
    combined = combined_turning_metrics(predictions, evaluations)

    predictions.to_csv(OUT_DIR / "predictions.csv", index=False, encoding="utf-8-sig")
    evaluations.to_csv(OUT_DIR / "evaluation.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(OUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    signals.to_csv(OUT_DIR / "signals.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": features}).to_csv(OUT_DIR / "features.csv", index=False, encoding="utf-8-sig")

    walk_low = evaluations[(evaluations["target"] == "is_low") & (evaluations["evaluation"] == "walk_forward")].iloc[0]
    walk_high = evaluations[(evaluations["target"] == "is_high") & (evaluations["evaluation"] == "walk_forward")].iloc[0]
    conclusion = (
        f"年度滚动口径下，低点精确召回率为 {walk_low['f2_recall']:.2%}，前后 3 日召回率为 "
        f"{walk_low['tolerance_recall']:.2%}；高点精确召回率为 {walk_high['f2_recall']:.2%}，前后 3 日召回率为 "
        f"{walk_high['tolerance_recall']:.2%}。全样本拟合与滚动结果的差距，代表未来信息不可得和市场状态变化造成的真实上限差异。"
    )
    summary = {
        "model": "V1ML",
        "label": "V1 original paired local extrema",
        "label_window": LABEL_WINDOW,
        "rows": int(len(panel)),
        "etfs": int(panel["symbol"].nunique()),
        "features": int(len(features)),
        "low_labels": int(panel["is_low"].sum()),
        "high_labels": int(panel["is_high"].sum()),
        "best_low_config": asdict(best_configs["is_low"]),
        "best_high_config": asdict(best_configs["is_high"]),
        "combined": combined,
        "evaluations": evaluation_rows,
        "conclusion": conclusion,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, evaluations, importance)
    print(json.dumps({"output": str(OUT_DIR), "conclusion": conclusion}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
