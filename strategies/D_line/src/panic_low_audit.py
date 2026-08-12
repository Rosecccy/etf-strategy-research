from __future__ import annotations

"""Search a causal panic-new-low add-on without changing the base low model."""

import json
from datetime import datetime

import numpy as np
import pandas as pd

from fit_dividend_extrema5 import OUT, PROJECT, SYMBOL, build_features


TUNE_YEARS = range(2013, 2020)
HOLDOUT_YEARS = range(2020, 2026)


def build_frame() -> pd.DataFrame:
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.dropna(subset=["date", "close", "high", "low", "volume"]).sort_values("date").reset_index(drop=True)
    features, labels = build_features(raw)
    aligned = raw.set_index("date").reindex(pd.to_datetime(labels["date"])).reset_index(drop=True)
    close = aligned["close"].astype(float)
    frame = pd.DataFrame({"date": pd.to_datetime(labels["date"]), "year": pd.to_datetime(labels["date"]).dt.year, "actual": labels["low"].astype(int)})
    frame["ret1"] = close.pct_change().to_numpy()
    frame["close_in_day"] = ((close - aligned["low"].astype(float)) / (aligned["high"].astype(float) - aligned["low"].astype(float)).replace(0, np.nan)).fillna(0.5).to_numpy()
    frame["week_rsi"] = features["W_RSI6"].to_numpy()
    frame["daily_rsi"] = features["D_RSI6"].to_numpy()
    frame["volume_ratio"] = features["volume_ratio_20"].to_numpy()
    for days in (5, 10, 20):
        prior_low = close.shift(1).rolling(days, min_periods=days).min()
        frame[f"new_low_{days}"] = (close < prior_low).to_numpy()
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def metric(frame: pd.DataFrame, mask: pd.Series) -> dict[str, float | int]:
    total = int(mask.sum()); hits = int(frame.loc[mask, "actual"].sum()); positives = int(frame["actual"].sum())
    annual = []
    for _, part in frame.groupby("year"):
        selected = mask.loc[part.index]
        annual.append(float(part.loc[selected, "actual"].mean()) if selected.any() else None)
    active = [value for value in annual if value is not None]
    return {"signals": total, "hits": hits, "precision": hits / total if total else 0.0, "recall": hits / positives if positives else 0.0, "min_year_precision": min(active) if active else 0.0, "active_years": len(active)}


def main() -> None:
    frame = build_frame()
    records = []
    for days in (5, 10, 20):
        for drop in (0.02, 0.03, 0.04, 0.05, 0.07, 0.09):
            for weekly_rsi in (30, 35, 40, 45):
                for daily_rsi in (20, 30, 40):
                    for min_volume in (0.0, 0.6, 1.0):
                        mask = (
                            frame[f"new_low_{days}"]
                            & (frame["ret1"] <= -drop)
                            & (frame["week_rsi"] <= weekly_rsi)
                            & (frame["daily_rsi"] <= daily_rsi)
                            & (frame["volume_ratio"] >= min_volume)
                        )
                        for stage, years in (("tune_2013_2019", TUNE_YEARS), ("holdout_2020_2025", HOLDOUT_YEARS)):
                            subset = frame.loc[frame["year"].isin(years)]
                            records.append({"stage": stage, "new_low_days": days, "drop": drop, "week_rsi_max": weekly_rsi, "daily_rsi_max": daily_rsi, "min_volume_ratio": min_volume, **metric(subset, mask.loc[subset.index])})
    all_rows = pd.DataFrame(records)
    tune = all_rows.loc[all_rows["stage"] == "tune_2013_2019"].copy()
    eligible = tune.loc[(tune["signals"] >= 12) & (tune["active_years"] >= 3)].copy()
    choice = eligible.sort_values(["precision", "min_year_precision", "hits", "recall"], ascending=False).iloc[0]
    holdout = all_rows.loc[(all_rows["stage"] == "holdout_2020_2025") & (all_rows["new_low_days"] == choice["new_low_days"]) & (all_rows["drop"] == choice["drop"]) & (all_rows["week_rsi_max"] == choice["week_rsi_max"]) & (all_rows["daily_rsi_max"] == choice["daily_rsi_max"]) & (all_rows["min_volume_ratio"] == choice["min_volume_ratio"])].iloc[0]
    result = {"new_low_days": int(choice["new_low_days"]), "drop": float(choice["drop"]), "week_rsi_max": float(choice["week_rsi_max"]), "daily_rsi_max": float(choice["daily_rsi_max"]), "min_volume_ratio": float(choice["min_volume_ratio"]), "tune_precision": float(choice["precision"]), "holdout_precision": float(holdout["precision"]), "holdout_recall": float(holdout["recall"]), "holdout_signals": int(holdout["signals"]), "active_years": int(holdout["active_years"]), "min_year_precision": float(holdout["min_year_precision"]), "validated": bool(holdout["precision"] >= 25 / 72 and holdout["active_years"] >= 3)}
    all_rows.to_csv(OUT / "panic_low_all.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([result]).to_csv(OUT / "panic_low_selected.csv", index=False, encoding="utf-8-sig")
    (OUT / "panic_low_summary.json").write_text(json.dumps({"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
