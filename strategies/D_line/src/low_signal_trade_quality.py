from __future__ import annotations

"""Compare exact labels, near-low hits and forward returns for low signals."""

import pandas as pd

from fit_dividend_extrema5 import OUT, PROJECT, SYMBOL


def add_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date").reset_index(drop=True)
    raw["close"] = raw["close"].astype(float)
    for days in (1, 3, 5, 10):
        raw[f"ret_{days}"] = raw["close"].shift(-days) / raw["close"] - 1.0
    raw["near_low_1"] = raw["date"].map(pd.Series(raw.index, index=raw["date"]))
    merged = frame.merge(raw[["date", "ret_1", "ret_3", "ret_5", "ret_10"]], on="date", how="left", validate="one_to_one")
    actual = merged["actual"].to_numpy()
    near = []
    for index in range(len(merged)):
        left = max(0, index - 1); right = min(len(merged), index + 2)
        near.append(bool(actual[left:right].max()))
    merged["near_low_1"] = near
    return merged


def summarize(frame: pd.DataFrame, mask: pd.Series, name: str) -> dict[str, float | int | str]:
    picks = frame.loc[mask].copy()
    row: dict[str, float | int | str] = {
        "strategy": name, "signals": len(picks), "exact_hits": int(picks["actual"].sum()),
        "exact_precision": float(picks["actual"].mean()) if len(picks) else 0.0,
        "near_hits_1day": int(picks["near_low_1"].sum()),
        "near_precision_1day": float(picks["near_low_1"].mean()) if len(picks) else 0.0,
    }
    for days in (1, 3, 5, 10):
        value = picks[f"ret_{days}"].dropna()
        row[f"avg_ret_{days}"] = float(value.mean()) if len(value) else 0.0
        row[f"win_rate_{days}"] = float((value > 0).mean()) if len(value) else 0.0
    return row


def main() -> None:
    base = pd.read_csv(OUT / "severity_score_predictions_2013_2025.csv", encoding="utf-8-sig")
    base["date"] = pd.to_datetime(base["date"])
    base = base.loc[base["year"].between(2020, 2025)].copy()
    base_quality = add_outcomes(base)
    base_mask = (base_quality["rank"] >= 0.90) & (base_quality["W_RSI6"] <= 45)
    report_rows = [summarize(base_quality, base_mask, "基线低点评分")]
    advanced_trades = []
    for window in (3, 5, 7, "all"):
        advanced = pd.read_csv(OUT / f"monotonic_leg_predictions_{999 if window == 'all' else window}.csv", encoding="utf-8-sig")
        advanced["date"] = pd.to_datetime(advanced["date"])
        advanced = advanced.loc[advanced["year"].between(2020, 2025)].copy()
        advanced_quality = add_outcomes(advanced)
        report_rows.append(summarize(advanced_quality, advanced_quality["selected_signal"], f"单调下跌补充分数（{window}年滚动）"))
        advanced_trades.append(advanced_quality.loc[advanced_quality["selected_signal"]])
    report = pd.DataFrame(report_rows)
    report.to_csv(OUT / "low_signal_trade_quality.csv", index=False, encoding="utf-8-sig")
    base_quality.loc[base_mask].to_csv(OUT / "low_signal_trade_quality_base_trades.csv", index=False, encoding="utf-8-sig")
    pd.concat(advanced_trades, ignore_index=True).to_csv(OUT / "low_signal_trade_quality_monotonic_trades.csv", index=False, encoding="utf-8-sig")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
