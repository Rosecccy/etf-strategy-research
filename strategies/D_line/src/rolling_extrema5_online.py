from __future__ import annotations

"""Strictly online, staged evaluation for 510880 five-day extrema fitting.

For a prediction year Y, the estimator is trained only through Y-2, its signal
threshold is calibrated only from scores in Y-1, and every date in Y is then
judged independently against that already-fixed threshold.  This intentionally
trades a little model freshness for a clean, reproducible no-look-ahead test.
"""

import json
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ablate_and_ensemble_dividend_extrema5 import groups
from fit_dividend_extrema5 import OUT, PROJECT, SYMBOL, WINDOW, build_features, scores
from tune_dividend_extrema5 import VARIANTS, model_for


TUNE_YEARS = range(2013, 2020)
HOLDOUT_YEARS = range(2020, 2026)
RATES = (0.025, 0.04, 0.055, 0.07, 0.085, 0.10, 0.125, 0.15)
# These four candidates are the diverse survivors of the earlier broad search:
# tree bagging, full-feature boosting, price-only boosting, and no-Fourier LGBM.
# Searching only survivors prevents a large parameter sweep from selecting noise.
CANDIDATES = (
    ("rf_d10_l3", "all"),
    ("hgb_leaf15_l2_4", "all"),
    ("hgb_leaf15_l2_4", "price_only"),
    ("lgb_leaf15_min20", "no_fourier"),
)
VARIANT_BY_NAME = {item[0]: item for item in VARIANTS}


@dataclass(frozen=True)
class Candidate:
    variant: str
    feature_group: str

    @property
    def name(self) -> str:
        return f"{self.variant}:{self.feature_group}"


def relative_rank(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return each value's percentile against an earlier, fixed reference set."""
    ordered = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(ordered, values, side="right") / len(ordered)


def threshold_metrics(frame: pd.DataFrame, rate: float) -> dict[str, float | int]:
    signal = frame["rank"] >= 1.0 - rate
    hits = int(frame.loc[signal, "actual"].sum())
    count = int(signal.sum())
    positives = int(frame["actual"].sum())
    precision = hits / count if count else 0.0
    recall = hits / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "rate": rate,
        "signals": count,
        "hits": hits,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def staged_predictions(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    target_name: str,
    columns: list[str],
    variant_name: str,
    years: range,
) -> pd.DataFrame:
    """Score years with no labels, scores, or thresholds from the future year."""
    dates = pd.to_datetime(labels["date"])
    target = labels[target_name].astype(int)
    rows: list[pd.DataFrame] = []
    frame = features[columns]
    for year in years:
        calibrate = dates.dt.year == year - 1
        test = dates.dt.year == year
        # A five-day extremum label at the end of Y-2 is not known on the
        # first date of Y-1 yet.  Remove that unresolved tail before fitting.
        calibration_start = np.flatnonzero(calibrate.to_numpy())
        if len(calibration_start) == 0:
            continue
        known_end = int(calibration_start[0]) - WINDOW
        train = pd.Series(False, index=features.index)
        train.iloc[:max(known_end, 0)] = True
        y_train = target.loc[train]
        if len(y_train) < 500 or y_train.sum() < 20 or not calibrate.any() or not test.any():
            continue
        model = model_for(VARIANT_BY_NAME[variant_name], float(y_train.mean()))
        model.fit(frame.loc[train], y_train)
        calibration_score = scores(model, frame.loc[calibrate])
        test_score = scores(model, frame.loc[test])
        rows.append(
            pd.DataFrame(
                {
                    "date": dates.loc[test].to_numpy(),
                    "year": year,
                    "actual": target.loc[test].to_numpy(),
                    "score": test_score,
                    "rank": relative_rank(test_score, calibration_score),
                    "candidate": f"{variant_name}:{next(name for name, value in FEATURE_GROUPS.items() if value == columns)}",
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def aggregate_metrics(predictions: pd.DataFrame, rates: tuple[float, ...] = RATES) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for rate in rates:
        yearly = []
        for _, fold in predictions.groupby("year"):
            yearly.append(threshold_metrics(fold, rate))
        combined = threshold_metrics(predictions, rate)
        rows.append(
            {
                **combined,
                "years": int(predictions["year"].nunique()),
                "mean_year_precision": float(np.mean([item["precision"] for item in yearly])),
                "mean_year_recall": float(np.mean([item["recall"] for item in yearly])),
                "min_year_precision": float(np.min([item["precision"] for item in yearly])),
                "mean_ap": float(
                    np.mean(
                        [average_precision_score(fold["actual"], fold["score"]) for _, fold in predictions.groupby("year")]
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def rank_candidates(records: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    # A point finder must recover enough true points as well as be selective.
    return frame.sort_values(
        ["f1", "precision", "recall", "min_year_precision", "mean_ap"],
        ascending=False,
    ).reset_index(drop=True)


def blend_predictions(predictions: list[pd.DataFrame], name: str) -> pd.DataFrame:
    """Average calibration-relative ranks; each component remains staged OOS."""
    joined = predictions[0][["date", "year", "actual", "rank"]].rename(columns={"rank": "rank_0"})
    for index, item in enumerate(predictions[1:], start=1):
        joined = joined.merge(
            item[["date", "rank"]].rename(columns={"rank": f"rank_{index}"}), on="date", how="inner"
        )
    rank_columns = [column for column in joined if column.startswith("rank_")]
    joined["rank"] = joined[rank_columns].mean(axis=1)
    joined["candidate"] = name
    joined["score"] = joined["rank"]
    return joined[["date", "year", "actual", "score", "rank", "candidate"]]


def evaluate_space(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    target_name: str,
    years: range,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    records: list[dict[str, object]] = []
    predictions_by_name: dict[str, pd.DataFrame] = {}
    for variant, group_name in CANDIDATES:
        candidate = Candidate(variant, group_name)
        prediction = staged_predictions(
            features, labels, target_name, FEATURE_GROUPS[group_name], variant, years
        )
        if prediction.empty:
            continue
        predictions_by_name[candidate.name] = prediction
        for row in aggregate_metrics(prediction).to_dict(orient="records"):
            records.append({"target": target_name, "kind": "single", "candidate": candidate.name, **row})
    ranked = rank_candidates(records)

    # Add only blends of the three independently tuned top base models.
    base_names = list(ranked.drop_duplicates("candidate").head(3)["candidate"])
    for size in (2, 3):
        members = base_names[:size]
        if len(members) != size:
            continue
        name = f"blend_{size}_equal(" + " | ".join(members) + ")"
        blend = blend_predictions([predictions_by_name[item] for item in members], name)
        predictions_by_name[name] = blend
        for row in aggregate_metrics(blend).to_dict(orient="records"):
            records.append({"target": target_name, "kind": "blend", "candidate": name, **row})
    return rank_candidates(records), predictions_by_name


def choose_and_validate(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    target_name: str,
    tuning: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    # Avoid a very sparse "winner" by requiring at least five annual signals on average.
    eligible = tuning.loc[tuning["signals"] >= tuning["years"] * 5].copy()
    chosen = eligible.iloc[0] if not eligible.empty else tuning.iloc[0]
    candidate_name = str(chosen["candidate"])
    if str(chosen["kind"]) == "single":
        variant, group_name = candidate_name.split(":", 1)
        holdout = staged_predictions(
            features, labels, target_name, FEATURE_GROUPS[group_name], variant, HOLDOUT_YEARS
        )
    else:
        content = candidate_name.split("(", 1)[1].rsplit(")", 1)[0]
        members = content.split(" | ")
        parts = []
        for member in members:
            variant, group_name = member.split(":", 1)
            parts.append(staged_predictions(features, labels, target_name, FEATURE_GROUPS[group_name], variant, HOLDOUT_YEARS))
        holdout = blend_predictions(parts, candidate_name)
    validated = aggregate_metrics(holdout)
    validated.insert(0, "target", target_name)
    validated.insert(1, "selected_candidate", candidate_name)
    return chosen, validated, holdout


def main() -> None:
    global FEATURE_GROUPS
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    features, labels = build_features(raw)
    all_groups = groups(features)
    FEATURE_GROUPS = {name: all_groups[name] for _, name in CANDIDATES}

    selected_rows: list[dict[str, object]] = []
    holdout_tables: list[pd.DataFrame] = []
    signal_tables: list[pd.DataFrame] = []
    for target_name in ("low", "high"):
        tuning, prediction_cache = evaluate_space(features, labels, target_name, TUNE_YEARS)
        tuning.to_csv(OUT / f"online_{target_name}_tuning_ranking.csv", index=False, encoding="utf-8-sig")
        chosen, holdout, signals = choose_and_validate(features, labels, target_name, tuning, prediction_cache)
        holdout_tables.append(holdout)
        rate = float(chosen["rate"])
        signals = signals.copy()
        signals["signal"] = (signals["rank"] >= 1.0 - rate).astype(int)
        signals["target"] = target_name
        signal_tables.append(signals)
        selected_rows.append(
            {
                "target": target_name,
                "candidate": str(chosen["candidate"]),
                "kind": str(chosen["kind"]),
                "rate": rate,
                "tune_f1": float(chosen["f1"]),
                "tune_precision": float(chosen["precision"]),
                "tune_recall": float(chosen["recall"]),
                "tune_signals": int(chosen["signals"]),
                "holdout_f1": float(holdout.loc[holdout["rate"] == rate, "f1"].iloc[0]),
                "holdout_precision": float(holdout.loc[holdout["rate"] == rate, "precision"].iloc[0]),
                "holdout_recall": float(holdout.loc[holdout["rate"] == rate, "recall"].iloc[0]),
                "holdout_signals": int(holdout.loc[holdout["rate"] == rate, "signals"].iloc[0]),
            }
        )
    selected = pd.DataFrame(selected_rows)
    holdout = pd.concat(holdout_tables, ignore_index=True)
    signals = pd.concat(signal_tables, ignore_index=True).sort_values(["date", "target"])
    selected.to_csv(OUT / "online_selected.csv", index=False, encoding="utf-8-sig")
    holdout.to_csv(OUT / "online_holdout_thresholds.csv", index=False, encoding="utf-8-sig")
    signals.to_csv(OUT / "online_holdout_signals_2020_2025.csv", index=False, encoding="utf-8-sig")
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": SYMBOL,
        "label": "Independent five-day extrema; near-equal extrema are merged within five trading days and 0.5% price tolerance.",
        "training_rule": "For prediction year Y, train through Y-2, calibrate the score threshold on Y-1, then predict every date of Y independently.",
        "tuning_period": "2013-2019",
        "untouched_holdout": "2020-2025",
        "selection_rule": "Maximize exact-point F1 subject to at least five signals per year; precision, recall, annual stability, then PR-AUC break ties.",
        "selected": selected.to_dict(orient="records"),
    }
    (OUT / "online_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
