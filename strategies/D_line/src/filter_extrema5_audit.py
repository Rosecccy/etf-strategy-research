from __future__ import annotations

"""Audit causal, interpretable filters on the strict five-day extrema signals.

The model, extremum labels and threshold are deliberately left unchanged.  A
filter only decides whether an already-triggered signal is eligible to trade.
All filters are selected on 2013-2019 and inspected once on 2020-2025.
"""

import json
from datetime import datetime

import pandas as pd

import rolling_extrema5_online as online
from ablate_and_ensemble_dividend_extrema5 import groups
from fit_dividend_extrema5 import OUT, PROJECT, SYMBOL, build_features


TUNE_YEARS = range(2013, 2020)
HOLDOUT_YEARS = range(2020, 2026)
LOW_CANDIDATE = "blend_3_equal(hgb_leaf15_l2_4:all | hgb_leaf15_l2_4:price_only | lgb_leaf15_min20:no_fourier)"
HIGH_CANDIDATE = "lgb_leaf15_min20:no_fourier"
BASE_RATE = 0.15


def metric(frame: pd.DataFrame, allowed: pd.Series) -> dict[str, float | int]:
    selected = frame.loc[(frame["rank"] >= 1.0 - BASE_RATE) & allowed]
    signals = len(selected)
    hits = int(selected["actual"].sum())
    positives = int(frame["actual"].sum())
    precision = hits / signals if signals else 0.0
    recall = hits / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"signals": signals, "hits": hits, "precision": precision, "recall": recall, "f1": f1}


def with_feature_rows(prediction: pd.DataFrame, features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    lookup = features.copy()
    lookup["date"] = pd.to_datetime(labels["date"]).to_numpy()
    return prediction.merge(lookup, on="date", how="left", validate="one_to_one")


def all_filters(target: str) -> list[tuple[str, callable]]:
    if target == "low":
        base = [
            ("无额外过滤", lambda f: pd.Series(True, index=f.index)),
            ("日RSI6≤30", lambda f: f["D_RSI6"] <= 30),
            ("日RSI6≤35", lambda f: f["D_RSI6"] <= 35),
            ("日RSI6≤40", lambda f: f["D_RSI6"] <= 40),
            ("日KDJ-J≤10", lambda f: f["D_KDJ_J"] <= 10),
            ("日KDJ-J≤20", lambda f: f["D_KDJ_J"] <= 20),
            ("日KDJ-J≤30", lambda f: f["D_KDJ_J"] <= 30),
            ("周RSI6≤40", lambda f: f["W_RSI6"] <= 40),
            ("周RSI6≤45", lambda f: f["W_RSI6"] <= 45),
            ("20日区间位置≤20%", lambda f: f["range_pos_20"] <= 0.20),
            ("20日区间位置≤30%", lambda f: f["range_pos_20"] <= 0.30),
            ("距20日高点回撤≥2%", lambda f: f["drawdown_20"] <= -0.02),
            ("距20日高点回撤≥4%", lambda f: f["drawdown_20"] <= -0.04),
            ("未跌破60日均线6%以上", lambda f: f["ma_ratio_60"] >= -0.06),
            ("量比20日在0.6至1.8", lambda f: f["volume_ratio_20"].between(0.6, 1.8)),
        ]
        combinations = [
            ("日RSI6≤40 + 周RSI6≤45", lambda f: (f["D_RSI6"] <= 40) & (f["W_RSI6"] <= 45)),
            ("日RSI6≤40 + 20日区间≤30%", lambda f: (f["D_RSI6"] <= 40) & (f["range_pos_20"] <= 0.30)),
            ("日KDJ-J≤30 + 20日区间≤30%", lambda f: (f["D_KDJ_J"] <= 30) & (f["range_pos_20"] <= 0.30)),
            ("日RSI6≤40 + 回撤≥2%", lambda f: (f["D_RSI6"] <= 40) & (f["drawdown_20"] <= -0.02)),
            ("日RSI6≤40 + 未深跌破60日线", lambda f: (f["D_RSI6"] <= 40) & (f["ma_ratio_60"] >= -0.06)),
            ("日RSI6≤40 + 量比正常", lambda f: (f["D_RSI6"] <= 40) & f["volume_ratio_20"].between(0.6, 1.8)),
            ("周RSI6≤45 + 未深跌破60日线", lambda f: (f["W_RSI6"] <= 45) & (f["ma_ratio_60"] >= -0.06)),
        ]
        return base + combinations
    base = [
        ("无额外过滤", lambda f: pd.Series(True, index=f.index)),
        ("日RSI6≥60", lambda f: f["D_RSI6"] >= 60),
        ("日RSI6≥65", lambda f: f["D_RSI6"] >= 65),
        ("日RSI6≥70", lambda f: f["D_RSI6"] >= 70),
        ("日KDJ-J≥70", lambda f: f["D_KDJ_J"] >= 70),
        ("日KDJ-J≥80", lambda f: f["D_KDJ_J"] >= 80),
        ("日KDJ-J≥90", lambda f: f["D_KDJ_J"] >= 90),
        ("周RSI6≥55", lambda f: f["W_RSI6"] >= 55),
        ("周RSI6≥60", lambda f: f["W_RSI6"] >= 60),
        ("20日区间位置≥80%", lambda f: f["range_pos_20"] >= 0.80),
        ("20日区间位置≥90%", lambda f: f["range_pos_20"] >= 0.90),
        ("20日量比≥1", lambda f: f["volume_ratio_20"] >= 1.0),
        ("20日均线上方", lambda f: f["ma_ratio_20"] >= 0),
    ]
    combinations = [
        ("日RSI6≥60 + 区间≥80%", lambda f: (f["D_RSI6"] >= 60) & (f["range_pos_20"] >= 0.80)),
        ("日RSI6≥65 + 区间≥80%", lambda f: (f["D_RSI6"] >= 65) & (f["range_pos_20"] >= 0.80)),
        ("日KDJ-J≥80 + 区间≥80%", lambda f: (f["D_KDJ_J"] >= 80) & (f["range_pos_20"] >= 0.80)),
        ("日RSI6≥60 + 周RSI6≥55", lambda f: (f["D_RSI6"] >= 60) & (f["W_RSI6"] >= 55)),
        ("日RSI6≥60 + 量比≥1", lambda f: (f["D_RSI6"] >= 60) & (f["volume_ratio_20"] >= 1.0)),
        ("区间≥80% + 量比≥1", lambda f: (f["range_pos_20"] >= 0.80) & (f["volume_ratio_20"] >= 1.0)),
        ("日RSI6≥60 + 20日均线上方", lambda f: (f["D_RSI6"] >= 60) & (f["ma_ratio_20"] >= 0)),
    ]
    return base + combinations


def candidate_predictions(features: pd.DataFrame, labels: pd.DataFrame, target: str, candidate: str, years: range) -> pd.DataFrame:
    def single(name: str) -> pd.DataFrame:
        variant, group_name = name.split(":", 1)
        return online.staged_predictions(features, labels, target, online.FEATURE_GROUPS[group_name], variant, years)

    if not candidate.startswith("blend"):
        return single(candidate)
    members = candidate.split("(", 1)[1].rsplit(")", 1)[0].split(" | ")
    return online.blend_predictions([single(member) for member in members], candidate)


def summarize(target: str, stage: str, frame: pd.DataFrame, specs: list[tuple[str, callable]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    yearly = []
    for name, func in specs:
        allowed = func(frame).fillna(False)
        total = metric(frame, allowed)
        years = []
        for year, fold in frame.groupby("year"):
            row = metric(fold, allowed.loc[fold.index])
            years.append(row)
            yearly.append({"target": target, "stage": stage, "filter": name, "year": int(year), **row})
        records.append({
            "target": target,
            "stage": stage,
            "filter": name,
            **total,
            "min_year_signals": min(row["signals"] for row in years),
            "min_year_precision": min(row["precision"] for row in years),
            "mean_year_precision": sum(row["precision"] for row in years) / len(years),
        })
    return pd.DataFrame(records), pd.DataFrame(yearly)


def main() -> None:
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    features, labels = build_features(raw)
    online.FEATURE_GROUPS = groups(features)
    selected = {"low": LOW_CANDIDATE, "high": HIGH_CANDIDATE}
    all_summary = []
    all_yearly = []
    winners = []
    for target, candidate in selected.items():
        specs = all_filters(target)
        tuning = with_feature_rows(candidate_predictions(features, labels, target, candidate, TUNE_YEARS), features, labels)
        tune_summary, tune_yearly = summarize(target, "tune_2013_2019", tuning, specs)
        # We select only filters which retain practical annual coverage.
        practical = tune_summary.loc[(tune_summary["signals"] >= 35) & (tune_summary["min_year_signals"] >= 2)].copy()
        winner = practical.sort_values(["precision", "min_year_precision", "recall", "f1"], ascending=False).iloc[0]
        holdout_all = pd.read_csv(OUT / "online_holdout_signals_2020_2025.csv", encoding="utf-8-sig")
        holdout = holdout_all.loc[(holdout_all["target"] == target) & (holdout_all["candidate"] == candidate)].copy()
        holdout["date"] = pd.to_datetime(holdout["date"])
        holdout = with_feature_rows(holdout, features, labels)
        # Audit all filters on the protected years for transparency, but only winner is a candidate.
        hold_summary, hold_yearly = summarize(target, "holdout_2020_2025", holdout, specs)
        all_summary.extend([tune_summary, hold_summary])
        all_yearly.extend([tune_yearly, hold_yearly])
        validated = hold_summary.loc[hold_summary["filter"] == winner["filter"]].iloc[0]
        baseline = hold_summary.loc[hold_summary["filter"] == "无额外过滤"].iloc[0]
        winners.append({
            "target": target,
            "candidate": candidate,
            "base_rate": BASE_RATE,
            "selected_filter_from_tune": winner["filter"],
            "tune_precision": winner["precision"],
            "tune_recall": winner["recall"],
            "tune_signals": int(winner["signals"]),
            "holdout_precision": validated["precision"],
            "holdout_recall": validated["recall"],
            "holdout_signals": int(validated["signals"]),
            "holdout_baseline_precision": baseline["precision"],
            "holdout_precision_change": validated["precision"] - baseline["precision"],
            "holdout_min_year_precision": validated["min_year_precision"],
            "validated": bool(validated["precision"] > baseline["precision"] and validated["min_year_signals"] >= 2),
        })
    summary = pd.concat(all_summary, ignore_index=True)
    yearly = pd.concat(all_yearly, ignore_index=True)
    selection = pd.DataFrame(winners)
    summary.to_csv(OUT / "filter_audit_all.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUT / "filter_audit_yearly.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(OUT / "filter_audit_selected.csv", index=False, encoding="utf-8-sig")
    payload = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "selection": winners}
    (OUT / "filter_audit_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
