from __future__ import annotations

"""Export the 2026 live candidates from the strictly staged extrema model."""

import json
from datetime import datetime

import numpy as np
import pandas as pd

from ablate_and_ensemble_dividend_extrema5 import groups
from fit_dividend_extrema5 import OUT, PROJECT, SYMBOL, WINDOW, build_features, scores
from rolling_extrema5_online import relative_rank
from tune_dividend_extrema5 import VARIANTS, model_for


TARGET_YEAR = 2026
VARIANT_BY_NAME = {item[0]: item for item in VARIANTS}


def members(candidate: str) -> list[tuple[str, str]]:
    if candidate.startswith("blend_"):
        candidate = candidate.split("(", 1)[1].rsplit(")", 1)[0]
        return [tuple(item.split(":", 1)) for item in candidate.split(" | ")]
    return [tuple(candidate.split(":", 1))]


def target_scores(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    target: str,
    candidate: str,
    rate: float,
    feature_groups: dict[str, list[str]],
) -> pd.DataFrame:
    dates = pd.to_datetime(labels["date"])
    calibrate = dates.dt.year == TARGET_YEAR - 1
    live = dates.dt.year == TARGET_YEAR
    first_calibration = np.flatnonzero(calibrate.to_numpy())
    if len(first_calibration) == 0 or not live.any():
        raise ValueError("The required calibration year or live year is unavailable.")
    # Same boundary as the evaluation: the prior year's early model knows only
    # labels confirmed before that calibration year began.
    known_end = int(first_calibration[0]) - WINDOW
    train = pd.Series(False, index=features.index)
    train.iloc[:max(known_end, 0)] = True
    y_train = labels.loc[train, target].astype(int)

    ranks = []
    for variant_name, group_name in members(candidate):
        frame = features[feature_groups[group_name]]
        model = model_for(VARIANT_BY_NAME[variant_name], float(y_train.mean()))
        model.fit(frame.loc[train], y_train)
        calibration_score = scores(model, frame.loc[calibrate])
        live_score = scores(model, frame.loc[live])
        ranks.append(relative_rank(live_score, calibration_score))
    rank = np.mean(np.vstack(ranks), axis=0)
    return pd.DataFrame(
        {
            "date": dates.loc[live].to_numpy(),
            f"{target}_score": rank,
            f"{target}_signal": (rank >= 1.0 - rate).astype(int),
        }
    )


def main() -> None:
    selection = pd.read_csv(OUT / "online_selected.csv", encoding="utf-8-sig")
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    features, labels = build_features(raw, include_unlabeled_tail=True)
    feature_groups = groups(features)

    panels = []
    selection_rows = []
    for target in ("low", "high"):
        item = selection.loc[selection["target"] == target].iloc[0]
        candidate = str(item["candidate"])
        rate = float(item["rate"])
        panels.append(target_scores(features, labels, target, candidate, rate, feature_groups))
        selection_rows.append({"target": target, "candidate": candidate, "rate": rate})
    panel = panels[0].merge(panels[1], on="date", how="outer")
    panel = panel.merge(raw[["date", "open", "high", "low", "close", "volume"]], on="date", how="left")
    panel = panel.sort_values("date").reset_index(drop=True)
    panel.to_csv(OUT / "online_live_predictions_2026.csv", index=False, encoding="utf-8-sig")

    latest = panel.iloc[-1]
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": SYMBOL,
        "data_latest_date": str(pd.Timestamp(latest["date"]).date()),
        "selection": selection_rows,
        "low_candidates_2026": int(panel["low_signal"].sum()),
        "high_candidates_2026": int(panel["high_signal"].sum()),
        "latest_low_score": float(latest["low_score"]),
        "latest_high_score": float(latest["high_score"]),
        "latest_low_candidate": bool(latest["low_signal"]),
        "latest_high_candidate": bool(latest["high_signal"]),
        "note": "Scores are percentiles against 2025 calibration scores, not calibrated probabilities. Signal thresholds and models are frozen before the 2026 live year.",
    }
    (OUT / "online_live_summary_2026.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
