from __future__ import annotations

"""Build transparent score-weighted OOS accounting and data for the extrema view."""

import json
from datetime import datetime

import numpy as np
import pandas as pd

from fit_dividend_extrema5 import OUT, PROJECT, SYMBOL


STARTING_CASH = 10_000.0
LOW_THRESHOLD = 0.85
HIGH_THRESHOLD = 0.85


def buy_fraction(score: float) -> float:
    """Use 20%-70% of currently available cash as low-point confidence rises."""
    return float(np.clip(0.20 + 0.50 * (score - LOW_THRESHOLD) / 0.15, 0.20, 0.70))


def sell_fraction(score: float) -> float:
    """Sell 30%-100% of held shares as high-point confidence rises."""
    return float(np.clip(0.30 + 0.70 * (score - HIGH_THRESHOLD) / 0.15, 0.30, 1.00))


def panel() -> pd.DataFrame:
    historical = pd.read_csv(OUT / "online_holdout_signals_2020_2025.csv", parse_dates=["date"])
    live = pd.read_csv(OUT / "online_live_predictions_2026.csv", parse_dates=["date"])
    historical = historical.pivot(index="date", columns="target", values=["score", "signal", "actual"])
    historical.columns = ["_".join(item) for item in historical.columns]
    historical = historical.reset_index()
    live = live.rename(
        columns={
            "low_score": "score_low",
            "high_score": "score_high",
            "low_signal": "signal_low",
            "high_signal": "signal_high",
        }
    )
    frame = pd.concat(
        [
            historical,
            live[["date", "score_low", "score_high", "signal_low", "signal_high"]],
        ],
        ignore_index=True,
    )
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig", parse_dates=["date"])
    raw = raw.sort_values("date").reset_index(drop=True)
    frame = raw.merge(frame, on="date", how="inner").sort_values("date").reset_index(drop=True)
    for column in ("score_low", "score_high", "signal_low", "signal_high", "actual_low", "actual_high"):
        frame[column] = pd.to_numeric(frame.get(column, 0), errors="coerce")
    frame[["signal_low", "signal_high"]] = frame[["signal_low", "signal_high"]].fillna(0).astype(int)
    frame["year"] = frame["date"].dt.year
    return frame


def forward_returns(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    oos = frame.loc[frame["year"].between(2020, 2025)].reset_index(drop=True)
    for target in ("low", "high"):
        signal_col = f"signal_{target}"
        score_col = f"score_{target}"
        for index in range(len(oos) - 21):
            row = oos.iloc[index]
            if not row[signal_col]:
                continue
            entry = float(oos.iloc[index + 1]["close"])
            for horizon in (5, 10, 20):
                rows.append(
                    {
                        "target": target,
                        "score": float(row[score_col]),
                        "horizon": horizon,
                        "return": float(oos.iloc[index + 1 + horizon]["close"] / entry - 1),
                    }
                )
    events = pd.DataFrame(rows)
    events["band"] = pd.cut(
        events["score"], [0.849, 0.90, 0.95, 1.001], labels=["85-90", "90-95", "95-100"], include_lowest=True
    )
    summary = (
        events.groupby(["target", "band", "horizon"], observed=False)["return"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    summary["band"] = summary["band"].astype(str)
    return summary


def score_weighted_account(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    oos = frame.loc[frame["year"].between(2020, 2025)].reset_index(drop=True)
    cash = STARTING_CASH
    shares = 0.0
    trades: list[dict[str, float | int | str]] = []
    curve: list[dict[str, float | int | str]] = []
    for index, row in oos.iterrows():
        if index:
            signal = oos.iloc[index - 1]
            # Signals are confirmed after close and executed at this next close.
            if signal["signal_high"]:
                fraction = sell_fraction(float(signal["score_high"]))
                quantity = shares * fraction
                if quantity > 1e-12:
                    value = quantity * float(row["close"])
                    shares -= quantity
                    cash += value
                    trades.append(
                        {
                            "signal_date": signal["date"].strftime("%Y-%m-%d"),
                            "exec_date": row["date"].strftime("%Y-%m-%d"),
                            "side": "sell",
                            "score": float(signal["score_high"]),
                            "fraction": fraction,
                            "price": float(row["close"]),
                            "value": value,
                        }
                    )
            if signal["signal_low"]:
                fraction = buy_fraction(float(signal["score_low"]))
                value = cash * fraction
                if value > 1e-12:
                    shares += value / float(row["close"])
                    cash -= value
                    trades.append(
                        {
                            "signal_date": signal["date"].strftime("%Y-%m-%d"),
                            "exec_date": row["date"].strftime("%Y-%m-%d"),
                            "side": "buy",
                            "score": float(signal["score_low"]),
                            "fraction": fraction,
                            "price": float(row["close"]),
                            "value": value,
                        }
                    )
        nav = cash + shares * float(row["close"])
        curve.append({"date": row["date"].strftime("%Y-%m-%d"), "nav": nav})
    buy_hold = STARTING_CASH / float(oos.iloc[0]["close"]) * float(oos.iloc[-1]["close"])
    trade_frame = pd.DataFrame(trades)
    summary = {
        "start": str(oos.iloc[0]["date"].date()),
        "end": str(oos.iloc[-1]["date"].date()),
        "starting_cash": STARTING_CASH,
        "final_nav": float(curve[-1]["nav"]),
        "return": float(curve[-1]["nav"] / STARTING_CASH - 1),
        "buy_hold_final": buy_hold,
        "buy_hold_return": float(buy_hold / STARTING_CASH - 1),
        "excess_return": float(curve[-1]["nav"] / buy_hold - 1),
        "trades": int(len(trade_frame)),
        "buys": int((trade_frame["side"] == "buy").sum()),
        "sells": int((trade_frame["side"] == "sell").sum()),
        "fees": 0.0,
        "rule": "T+1 close execution; buy 20%-70% of current cash by low score; sell 30%-100% of held shares by high score; sell first if both occur; no fees or slippage.",
    }
    return trade_frame, {**summary, "curve": curve}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = panel()
    forward = forward_returns(data)
    trades, account = score_weighted_account(data)
    trades.to_csv(OUT / "score_weighted_oos_trades.csv", index=False, encoding="utf-8-sig")
    forward.to_csv(OUT / "score_forward_returns.csv", index=False, encoding="utf-8-sig")
    (OUT / "score_weighted_oos_summary.json").write_text(
        json.dumps({key: value for key, value in account.items() if key != "curve"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    visual = {
        "symbol": SYMBOL,
        "data": [
            {
                "d": row.date.strftime("%Y-%m-%d"),
                "c": round(float(row.close), 4),
                "ls": None if pd.isna(row.score_low) else round(float(row.score_low), 4),
                "hs": None if pd.isna(row.score_high) else round(float(row.score_high), 4),
                "lb": int(row.signal_low),
                "hb": int(row.signal_high),
                "al": None if pd.isna(row.actual_low) else int(row.actual_low),
                "ah": None if pd.isna(row.actual_high) else int(row.actual_high),
            }
            for row in data.itertuples(index=False)
        ],
        "forward": forward.to_dict(orient="records"),
        "account": account,
    }
    (OUT / "online_visual_data.json").write_text(json.dumps(visual, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({key: value for key, value in account.items() if key != "curve"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
