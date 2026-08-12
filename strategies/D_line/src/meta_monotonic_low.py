from __future__ import annotations

"""A strictly staged monotonic meta-model for five-day low-point scores."""

import json
from datetime import datetime

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

import rolling_extrema5_online as online
from fit_dividend_extrema5 import OUT, PROJECT, SYMBOL
from rolling_monotonic_descent import add_leg_state
from shape_low_model import shape_features


TUNE_YEARS = range(2017, 2020)
HOLDOUT_YEARS = range(2020, 2026)
ALL_YEARS = range(2016, 2026)
RATES = (0.055, 0.07, 0.085, 0.10, 0.125)
MODELS = ((3, 30), (7, 30), (7, 60), (15, 60))


def matrix(frame: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    source = raw.set_index("date").reindex(pd.to_datetime(frame["date"])).reset_index(drop=True)
    close = source["close"].astype(float)
    ret = close.pct_change()
    falling = ret.lt(0)
    out = pd.DataFrame(index=frame.index)
    out["base_rank"] = frame["rank"].astype(float)
    out["severity"] = frame["severity"].astype(float)
    out["leg_depth"] = (-frame["leg_drop"].astype(float)).clip(0.0, 0.30).fillna(0.0)
    out["leg_new_low"] = frame["leg_new_low"].astype(float)
    out["new_low_5"] = (close < close.shift(1).rolling(5, min_periods=5).min()).astype(float)
    out["new_low_10"] = (close < close.shift(1).rolling(10, min_periods=10).min()).astype(float)
    out["down_streak"] = falling.groupby((~falling).cumsum()).cumsum().clip(0, 15).astype(float)
    out["oversold_daily"] = -frame["D_RSI6"].astype(float)
    out["oversold_weekly"] = -frame["W_RSI6"].astype(float)
    out["near_range_floor"] = -frame["range_pos_20"].astype(float)
    out["drawdown_depth"] = -frame["drawdown_20"].astype(float)
    columns = out.columns.tolist()
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0), columns


def staged(frame: pd.DataFrame, features: pd.DataFrame, columns: list[str], leaves: int, min_child: int) -> pd.DataFrame:
    rows = []
    dates = pd.to_datetime(frame["date"])
    for year in ALL_YEARS:
        train = dates.dt.year <= year - 2
        calibrate = dates.dt.year == year - 1
        test = dates.dt.year == year
        if train.sum() < 500 or frame.loc[train, "actual"].sum() < 20 or not calibrate.any() or not test.any():
            continue
        y_train = frame.loc[train, "actual"].astype(int)
        positive_weight = max((1 - y_train.mean()) / y_train.mean(), 1.0)
        model = LGBMClassifier(
            objective="binary", n_estimators=240, learning_rate=0.03, num_leaves=leaves,
            max_depth=5, min_child_samples=min_child, colsample_bytree=0.9, subsample=0.9,
            reg_lambda=4.0, reg_alpha=0.3, scale_pos_weight=positive_weight,
            monotone_constraints=[1] * len(columns), random_state=17, verbosity=-1, n_jobs=1,
        )
        model.fit(features.loc[train, columns], y_train)
        calibration_score = model.predict_proba(features.loc[calibrate, columns])[:, 1]
        test_score = model.predict_proba(features.loc[test, columns])[:, 1]
        rows.append(pd.DataFrame({
            "date": dates.loc[test].to_numpy(), "year": year, "actual": frame.loc[test, "actual"].to_numpy(),
            "score": test_score, "rank": online.relative_rank(test_score, calibration_score),
            "base_rank": frame.loc[test, "rank"].to_numpy(), "leg_depth": features.loc[test, "leg_depth"].to_numpy(),
            "leg_new_low": features.loc[test, "leg_new_low"].to_numpy(), "W_RSI6": frame.loc[test, "W_RSI6"].to_numpy(),
            "D_RSI6": frame.loc[test, "D_RSI6"].to_numpy(), "leaves": leaves, "min_child": min_child,
        }))
    return pd.concat(rows, ignore_index=True)


def evaluate(frame: pd.DataFrame, stage: str, name: str) -> pd.DataFrame:
    rows = []
    for rate in RATES:
        selected = (frame["rank"] >= 1 - rate) & (frame["W_RSI6"] <= 45)
        total = int(selected.sum()); hits = int(frame.loc[selected, "actual"].sum()); positives = int(frame["actual"].sum())
        annual = []
        for _, part in frame.groupby("year"):
            mask = selected.loc[part.index]
            annual.append(float(part.loc[mask, "actual"].mean()) if mask.any() else 0.0)
        rows.append({"stage": stage, "candidate": name, "rate": rate, "signals": total, "hits": hits, "precision": hits / total if total else 0.0, "recall": hits / positives if positives else 0.0, "min_year_precision": min(annual)})
    return pd.DataFrame(rows)


def main() -> None:
    base = pd.read_csv(OUT / "severity_score_predictions_2013_2025.csv", encoding="utf-8-sig")
    base["date"] = pd.to_datetime(base["date"])
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"])
    base = base.merge(raw[["date", "close"]], on="date", how="left", validate="one_to_one")
    base["base_signal"] = (base["rank"] >= 0.90) & (base["W_RSI6"] <= 45)
    state = add_leg_state(base, 10)
    # Indicators already exist in the original feature table; merge the few
    # values needed by the monotonic meta learner without using future labels.
    from fit_dividend_extrema5 import build_features
    original_features, labels = build_features(raw)
    original_features = original_features.assign(date=pd.to_datetime(labels["date"]))
    state = state.merge(original_features[["date", "D_RSI6", "range_pos_20", "drawdown_20"]], on="date", how="left", validate="one_to_one")
    features, columns = matrix(state, raw)

    cache = {}; tuning = []
    for leaves, min_child in MODELS:
        name = f"lgb_mono_leaf{leaves}_min{min_child}"
        pred = staged(state, features, columns, leaves, min_child)
        cache[name] = pred
        tuning.append(evaluate(pred.loc[pred["year"].isin(TUNE_YEARS)], "tune_2017_2019", name))
    tuning_table = pd.concat(tuning, ignore_index=True)
    eligible = tuning_table.loc[(tuning_table["signals"] >= 15) & (tuning_table["min_year_precision"] > 0)].copy()
    choice = eligible.sort_values(["precision", "min_year_precision", "recall"], ascending=False).iloc[0]
    chosen = cache[str(choice["candidate"])]
    holdout = evaluate(chosen.loc[chosen["year"].isin(HOLDOUT_YEARS)], "holdout_2020_2025", str(choice["candidate"]))
    held = holdout.loc[holdout["rate"] == choice["rate"]].iloc[0]
    result = {
        "candidate": str(choice["candidate"]), "rate": float(choice["rate"]), "tune_precision": float(choice["precision"]),
        "holdout_precision": float(held["precision"]), "holdout_recall": float(held["recall"]), "holdout_signals": int(held["signals"]),
        "baseline_precision": 25 / 72, "precision_change": float(held["precision"] - 25 / 72),
        "min_year_precision": float(held["min_year_precision"]), "validated": bool(held["precision"] > 25 / 72 and held["min_year_precision"] > 0),
    }
    tuning_table.to_csv(OUT / "meta_mono_tuning.csv", index=False, encoding="utf-8-sig")
    holdout.to_csv(OUT / "meta_mono_holdout.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([result]).to_csv(OUT / "meta_mono_selected.csv", index=False, encoding="utf-8-sig")
    chosen.to_csv(OUT / "meta_mono_predictions_2016_2025.csv", index=False, encoding="utf-8-sig")
    (OUT / "meta_mono_summary.json").write_text(json.dumps({"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
