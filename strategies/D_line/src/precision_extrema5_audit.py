from __future__ import annotations

"""Choose precision-first extrema settings on tuning years, then audit holdout years."""

import json
from datetime import datetime

import pandas as pd

import rolling_extrema5_online as online
from ablate_and_ensemble_dividend_extrema5 import groups
from fit_dividend_extrema5 import OUT, PROJECT, SYMBOL, build_features


TUNE_YEARS = online.TUNE_YEARS
HOLDOUT_YEARS = online.HOLDOUT_YEARS
MIN_SIGNALS = len(TUNE_YEARS) * 5


def parse_members(candidate: str) -> list[tuple[str, str]]:
    if candidate.startswith("blend_"):
        content = candidate.split("(", 1)[1].rsplit(")", 1)[0]
        return [tuple(item.split(":", 1)) for item in content.split(" | ")]
    return [tuple(candidate.split(":", 1))]


def select_rows(tuning: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    base = tuning.loc[tuning["signals"] >= MIN_SIGNALS].copy()
    stable = base.loc[base["min_year_precision"] > 0].copy()
    choices = [
        ("平衡F1", base.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0]),
        ("最高精确率", base.sort_values(["precision", "recall", "f1"], ascending=False).iloc[0]),
        (
            "稳定精确率",
            (stable if not stable.empty else base).sort_values(
                ["precision", "min_year_precision", "recall"], ascending=False
            ).iloc[0],
        ),
    ]
    unique: dict[tuple[str, float], tuple[str, pd.Series]] = {}
    for label, row in choices:
        unique.setdefault((str(row["candidate"]), float(row["rate"])), (label, row))
    return list(unique.values())


def staged_candidate(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    target: str,
    candidate: str,
    cache: dict[tuple[str, str, str], pd.DataFrame],
) -> pd.DataFrame:
    pieces = []
    for variant, group_name in parse_members(candidate):
        key = (target, variant, group_name)
        if key not in cache:
            cache[key] = online.staged_predictions(
                features, labels, target, online.FEATURE_GROUPS[group_name], variant, HOLDOUT_YEARS
            )
        pieces.append(cache[key])
    if len(pieces) == 1:
        return pieces[0]
    return online.blend_predictions(pieces, candidate)


def main() -> None:
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    features, labels = build_features(raw)
    online.FEATURE_GROUPS = groups(features)
    cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []
    yearly_rows: list[pd.DataFrame] = []
    for target in ("low", "high"):
        tuning = pd.read_csv(OUT / f"online_{target}_tuning_ranking.csv")
        for choice_label, row in select_rows(tuning):
            candidate = str(row["candidate"])
            rate = float(row["rate"])
            holdout = staged_candidate(features, labels, target, candidate, cache)
            metrics = online.threshold_metrics(holdout, rate)
            yearly = []
            for year, fold in holdout.groupby("year"):
                yearly.append({"year": int(year), **online.threshold_metrics(fold, rate)})
            yearly_frame = pd.DataFrame(yearly)
            yearly_frame.insert(0, "target", target)
            yearly_frame.insert(1, "choice", choice_label)
            yearly_frame.insert(2, "candidate", candidate)
            yearly_rows.append(yearly_frame)
            summary_rows.append(
                {
                    "target": target,
                    "choice": choice_label,
                    "candidate": candidate,
                    "rate": rate,
                    "tune_precision": float(row["precision"]),
                    "tune_recall": float(row["recall"]),
                    "tune_f1": float(row["f1"]),
                    "tune_signals": int(row["signals"]),
                    "holdout_precision": float(metrics["precision"]),
                    "holdout_recall": float(metrics["recall"]),
                    "holdout_f1": float(metrics["f1"]),
                    "holdout_signals": int(metrics["signals"]),
                    "holdout_min_year_precision": float(yearly_frame["precision"].min()),
                }
            )
    summary = pd.DataFrame(summary_rows)
    yearly = pd.concat(yearly_rows, ignore_index=True)
    summary.to_csv(OUT / "precision_audit_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUT / "precision_audit_yearly.csv", index=False, encoding="utf-8-sig")
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rule": "Select only on 2013-2019. Require at least five tuning-year signals per year. Re-evaluate the selected setting only on untouched 2020-2025.",
        "results": summary.to_dict(orient="records"),
    }
    (OUT / "precision_audit_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
