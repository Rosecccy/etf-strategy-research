from __future__ import annotations

"""Causal model search for independent 5-day extrema labels on 510880."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

import factor_dca_scan as dca


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
OUT = ROOT / "out" / "extrema5"
SYMBOL = "510880"
WINDOW = 5
TOLERANCE = 0.005
MAX_GAP = 5
FIRST_OOS_YEAR = 2013
LAST_COMPLETE_OOS_YEAR = 2025


@dataclass(frozen=True)
class ModelSpec:
    name: str
    feature_set: str


ALL_MODELS = (
    ModelSpec("logit_all", "all"),
    ModelSpec("poly2_logit", "poly"),
    ModelSpec("fourier_logit", "fourier"),
    ModelSpec("random_forest", "all"),
    ModelSpec("extra_trees", "all"),
    ModelSpec("hist_gradient", "all"),
    ModelSpec("lightgbm", "all"),
    ModelSpec("rbf_svm", "all"),
)


def causal_fft(values: pd.Series, window: int, harmonic: int) -> pd.Series:
    source = values.to_numpy(dtype=float)
    result = np.full(len(source), np.nan)
    for end in range(window - 1, len(source)):
        segment = source[end - window + 1 : end + 1]
        if not np.all(np.isfinite(segment)):
            continue
        centered = segment - segment.mean()
        spectrum = np.abs(np.fft.rfft(centered))
        denominator = spectrum[1:].sum()
        if len(spectrum) > harmonic and denominator > 0:
            result[end] = spectrum[harmonic] / denominator
    return pd.Series(result, index=values.index)


def build_features(raw: pd.DataFrame, include_unlabeled_tail: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = raw.copy().sort_values("date").reset_index(drop=True)
    daily = dca.indicators(data)
    features = pd.DataFrame(index=data.index)

    for frequency in ("D", "W", "B15", "M"):
        state = dca.to_daily_state(data, frequency, activate_on_next_daily=False)
        for column in (*dca.NUMERIC_FACTORS.keys(), "BOLL_LOW", "BOLL_HIGH"):
            features[f"{frequency}_{column}"] = pd.to_numeric(state[column], errors="coerce")

    close = pd.to_numeric(data["close"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    volume = pd.to_numeric(data["volume"], errors="coerce").replace(0, np.nan)
    ret1 = close.pct_change()
    for period in (1, 2, 3, 5, 10, 20, 40, 60):
        features[f"ret_{period}"] = close.pct_change(period)
    for period in (5, 10, 20, 60, 120):
        moving = close.rolling(period, min_periods=period).mean()
        features[f"ma_ratio_{period}"] = close / moving - 1
        features[f"ma_slope_{period}"] = moving.pct_change(5)
    for period in (5, 10, 20, 60):
        rolling_low = low.rolling(period, min_periods=period).min()
        rolling_high = high.rolling(period, min_periods=period).max()
        features[f"range_pos_{period}"] = (close - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)
        features[f"realized_vol_{period}"] = ret1.rolling(period, min_periods=period).std()
        features[f"drawdown_{period}"] = close / close.rolling(period, min_periods=period).max() - 1
    features["intraday_range"] = (high - low) / close.replace(0, np.nan)
    features["volume_ratio_5"] = volume / volume.rolling(5, min_periods=5).mean()
    features["volume_ratio_20"] = volume / volume.rolling(20, min_periods=20).mean()
    features["volume_z_20"] = (np.log1p(volume) - np.log1p(volume).rolling(20, min_periods=20).mean()) / np.log1p(volume).rolling(20, min_periods=20).std()
    features["fft20_h1"] = causal_fft(ret1, 20, 1)
    features["fft20_h2"] = causal_fft(ret1, 20, 2)
    features["fft40_h1"] = causal_fft(ret1, 40, 1)
    features["fft40_h2"] = causal_fft(ret1, 40, 2)

    lows, highs = dca.clustered_extrema_labels(close, window=WINDOW, tolerance=TOLERANCE, max_gap=MAX_GAP)
    labels = pd.DataFrame({"date": data["date"], "low": lows.astype(int), "high": highs.astype(int)})
    # The last five closes cannot have a complete future window for label checking.
    valid = pd.Series(False, index=data.index)
    end = len(data) if include_unlabeled_tail else len(data) - WINDOW
    valid.iloc[120:end] = True
    features = features.replace([np.inf, -np.inf], np.nan)
    return features.loc[valid].copy(), labels.loc[valid].copy()


def feature_sets(features: pd.DataFrame) -> dict[str, list[str]]:
    poly = [
        "D_KDJ_K", "D_KDJ_D", "D_KDJ_J", "D_KD", "D_RSI6", "D_DIF", "D_EMV", "D_LON",
        "ret_1", "ret_5", "ret_10", "ret_20", "ma_ratio_5", "ma_ratio_20", "ma_ratio_60",
        "range_pos_5", "range_pos_20", "drawdown_20", "realized_vol_20", "volume_ratio_20",
    ]
    fourier = [column for column in features.columns if column.startswith("fft")]
    fourier += ["ret_1", "ret_5", "ret_10", "ret_20", "range_pos_5", "range_pos_20", "D_KDJ_J", "D_RSI6", "D_LON"]
    return {
        "all": list(features.columns),
        "poly": [column for column in poly if column in features],
        "fourier": [column for column in fourier if column in features],
    }


def make_model(name: str, positive_rate: float) -> object:
    weight = max((1 - positive_rate) / max(positive_rate, 1e-4), 1.0)
    if name == "logit_all":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.15, class_weight="balanced", max_iter=3000)),
        ])
    if name == "poly2_logit":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", LogisticRegression(C=0.015, class_weight="balanced", max_iter=3000)),
        ])
    if name == "fourier_logit":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.35, class_weight="balanced", max_iter=3000)),
        ])
    if name == "random_forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=400, max_depth=7, min_samples_leaf=4, max_features="sqrt",
                class_weight="balanced_subsample", random_state=7, n_jobs=-1,
            )),
        ])
    if name == "extra_trees":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=450, max_depth=8, min_samples_leaf=3, max_features="sqrt",
                class_weight="balanced", random_state=7, n_jobs=-1,
            )),
        ])
    if name == "hist_gradient":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0,
                class_weight={0: 1.0, 1: weight}, random_state=7,
            )),
        ])
    if name == "lightgbm":
        return LGBMClassifier(
            objective="binary", n_estimators=260, learning_rate=0.035, num_leaves=15,
            max_depth=5, min_child_samples=25, subsample=0.85, colsample_bytree=0.8,
            reg_lambda=2.0, reg_alpha=0.2, scale_pos_weight=weight, random_state=7,
            verbosity=-1, n_jobs=-1,
        )
    if name == "rbf_svm":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", SVC(C=0.8, gamma="scale", class_weight="balanced", kernel="rbf")),
        ])
    raise ValueError(name)


def scores(model: object, frame: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(frame)[:, 1]
    return model.decision_function(frame)


def event_metrics(target: np.ndarray, score: np.ndarray, expected_rate: float) -> dict[str, float | int]:
    if len(np.unique(target)) < 2:
        return {"ap": np.nan, "roc_auc": np.nan, "top_precision": np.nan, "top_recall": np.nan, "signals": 0}
    count = max(1, int(round(len(target) * expected_rate)))
    selected = np.argsort(score)[-count:]
    hits = int(target[selected].sum())
    positives = int(target.sum())
    return {
        "ap": float(average_precision_score(target, score)),
        "roc_auc": float(roc_auc_score(target, score)),
        "top_precision": float(hits / count),
        "top_recall": float(hits / positives) if positives else np.nan,
        "signals": count,
    }


def run_target(features: pd.DataFrame, labels: pd.DataFrame, target_name: str, sets: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    target = labels[target_name].to_numpy(dtype=int)
    dates = pd.to_datetime(labels["date"])
    rows: list[dict] = []
    yearly: list[dict] = []
    for spec in ALL_MODELS:
        columns = sets[spec.feature_set]
        frame = features[columns]
        fit_model = make_model(spec.name, float(target.mean()))
        fit_model.fit(frame, target)
        fit_stats = event_metrics(target, scores(fit_model, frame), float(target.mean()))

        fold_metrics: list[dict] = []
        for year in range(FIRST_OOS_YEAR, LAST_COMPLETE_OOS_YEAR + 1):
            train_mask = dates.dt.year < year
            test_mask = dates.dt.year == year
            y_train = target[train_mask]
            y_test = target[test_mask]
            if len(y_train) < 500 or y_train.sum() < 20 or y_test.sum() < 2:
                continue
            model = make_model(spec.name, float(y_train.mean()))
            model.fit(frame.loc[train_mask], y_train)
            metrics = event_metrics(y_test, scores(model, frame.loc[test_mask]), float(y_train.mean()))
            fold = {"target": target_name, "model": spec.name, "feature_set": spec.feature_set, "year": year, **metrics}
            yearly.append(fold)
            fold_metrics.append(fold)
        fold_frame = pd.DataFrame(fold_metrics)
        if fold_frame.empty:
            continue
        rows.append(
            {
                "target": target_name,
                "model": spec.name,
                "feature_set": spec.feature_set,
                "features": len(columns),
                "fit_ap": fit_stats["ap"],
                "fit_roc_auc": fit_stats["roc_auc"],
                "fit_top_precision": fit_stats["top_precision"],
                "oos_years": len(fold_frame),
                "oos_ap": fold_frame["ap"].mean(),
                "oos_roc_auc": fold_frame["roc_auc"].mean(),
                "oos_top_precision": fold_frame["top_precision"].mean(),
                "oos_top_recall": fold_frame["top_recall"].mean(),
                "oos_signals": int(fold_frame["signals"].sum()),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(["oos_ap", "oos_top_precision", "fit_ap"], ascending=False)
    best = ranking.iloc[0].to_dict() if not ranking.empty else {}
    return ranking, pd.DataFrame(yearly), best


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    features, labels = build_features(raw)
    sets = feature_sets(features)

    rankings: list[pd.DataFrame] = []
    yearly: list[pd.DataFrame] = []
    best: dict[str, object] = {}
    for target_name in ("low", "high"):
        ranking, folds, chosen = run_target(features, labels, target_name, sets)
        rankings.append(ranking)
        yearly.append(folds)
        best[target_name] = chosen

    ranking_frame = pd.concat(rankings, ignore_index=True)
    fold_frame = pd.concat(yearly, ignore_index=True)
    ranking_frame.to_csv(OUT / "model_ranking.csv", index=False, encoding="utf-8-sig")
    fold_frame.to_csv(OUT / "yearly_oos_metrics.csv", index=False, encoding="utf-8-sig")
    feature_snapshot = features.copy()
    feature_snapshot.insert(0, "date", labels["date"].to_numpy())
    feature_snapshot["label_low"] = labels["low"].to_numpy()
    feature_snapshot["label_high"] = labels["high"].to_numpy()
    feature_snapshot.to_csv(OUT / "feature_label_panel.csv", index=False, encoding="utf-8-sig")
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": SYMBOL,
        "name": "红利ETF",
        "label_definition": "Independent 5-day local extrema; near-equal extrema within five trading days and 0.5% price tolerance are merged. No buy/sell pairing is used.",
        "causal_features": "All indicators and Fourier features use data available at that close or earlier. Weekly, 15-day, and monthly values appear on each completed period-end close.",
        "evaluation": "Expanding yearly rolling OOS, 2013-2025. Scores are ranked by mean annual PR-AUC, then top-signal precision. In-sample fit is reported separately and is not a live-performance claim.",
        "samples": int(len(labels)),
        "low_labels": int(labels["low"].sum()),
        "high_labels": int(labels["high"].sum()),
        "feature_count": int(features.shape[1]),
        "best": best,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    shown = ranking_frame.copy()
    for column in ("fit_ap", "fit_roc_auc", "fit_top_precision", "oos_ap", "oos_roc_auc", "oos_top_precision", "oos_top_recall"):
        shown[column] = shown[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "")
    report = [
        "# 红利ETF 5日阶段高低点拟合",
        "",
        "标签为事后定义，用于监督学习；模型特征仅使用当日收盘时已知数据。拟合分数与滚动样本外分数必须分开看。",
        "",
        f"- 样本：{summary['samples']}日；低点：{summary['low_labels']}个；高点：{summary['high_labels']}个；特征：{summary['feature_count']}个。",
        f"- 样本外区间：{FIRST_OOS_YEAR}-{LAST_COMPLETE_OOS_YEAR}，逐年扩展训练。",
        "- `OOS PR-AUC`越高，代表模型越能把真实高/低点排在前面；`OOS Top Precision`表示按过去标签密度发出信号时，信号命中真实点的比例。",
        "",
        "## 模型排名",
        shown.to_markdown(index=False),
    ]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(ranking_frame.to_string(index=False))


if __name__ == "__main__":
    main()
