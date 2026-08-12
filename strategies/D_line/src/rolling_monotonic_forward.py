from __future__ import annotations

"""Choose monotonic low-score parameters by prior five-day trade quality."""

import json
from datetime import datetime

import pandas as pd

from fit_dividend_extrema5 import OUT, PROJECT, SYMBOL
from rolling_monotonic_descent import add_leg_state, config_grid, config_key, metrics, prepare_candidates


YEARS = range(2020, 2026)
WINDOWS = (3, 5, 7, 999)


def add_returns(frame: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    prices = raw[["date", "close"]].copy().sort_values("date")
    prices["ret5"] = prices["close"].shift(-5) / prices["close"] - 1.0
    return frame.merge(prices[["date", "ret5"]], on="date", how="left", validate="one_to_one")


def choose(history: pd.DataFrame, grid: list[dict], cache: dict[str, pd.DataFrame]) -> dict:
    candidates = []
    for config in grid:
        part = cache[config_key(config)].loc[history["date"]].reset_index()
        trades = part.loc[part["selected_signal"] & part["ret5"].notna(), "ret5"]
        if len(trades) < 15:
            continue
        candidates.append({**config, "trade_count": len(trades), "mean_ret5": float(trades.mean()), "win_rate5": float((trades > 0).mean()), "median_ret5": float(trades.median())})
    table = pd.DataFrame(candidates)
    # Median return breaks ties so one exceptional crash rebound cannot decide a rule.
    return table.sort_values(["mean_ret5", "median_ret5", "win_rate5", "trade_count"], ascending=False).iloc[0].to_dict()


def outcome(frame: pd.DataFrame) -> dict:
    chosen = frame["selected_signal"]
    stat = metrics(frame, chosen)
    trades = frame.loc[chosen & frame["ret5"].notna(), "ret5"]
    stat.update({"mean_ret5": float(trades.mean()), "median_ret5": float(trades.median()), "win_rate5": float((trades > 0).mean()), "trade_count": len(trades)})
    return stat


def main() -> None:
    base = pd.read_csv(OUT / "severity_score_predictions_2013_2025.csv", encoding="utf-8-sig")
    base["date"] = pd.to_datetime(base["date"])
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"])
    base = base.merge(raw[["date", "close"]], on="date", how="left", validate="one_to_one")
    base["base_signal"] = (base["rank"] >= 0.90) & (base["W_RSI6"] <= 45)
    states = {lookback: add_leg_state(base, lookback) for lookback in (3, 5, 7, 10)}
    grid = config_grid(states)
    cache = prepare_candidates(base, states, grid)
    cache = {key: add_returns(value.reset_index(), raw).set_index("date") for key, value in cache.items()}

    summaries = []; choices = []
    for window in WINDOWS:
        predictions = []
        for year in YEARS:
            first = int(base["year"].min()) if window == 999 else max(int(base["year"].min()), year - window)
            history = base.loc[(base["year"] >= first) & (base["year"] < year)]
            config = choose(history, grid, cache)
            test_dates = base.loc[base["year"] == year, "date"]
            test = cache[config_key(config)].loc[test_dates].reset_index().copy()
            test["history_window"] = "all" if window == 999 else window
            for key, value in config.items():
                test[f"selected_{key}"] = value
            predictions.append(test)
            choices.append({"window": "all" if window == 999 else window, "year": year, "history_start": first, "history_end": year - 1, **config, **outcome(test)})
        combined = pd.concat(predictions, ignore_index=True)
        record = outcome(combined)
        record["window"] = "all" if window == 999 else window
        summaries.append(record)
        combined.to_csv(OUT / f"monotonic_forward_predictions_{window}.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(summaries).sort_values(["mean_ret5", "median_ret5", "precision"], ascending=False)
    pd.DataFrame(choices).to_csv(OUT / "monotonic_forward_choices.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "monotonic_forward_summary.csv", index=False, encoding="utf-8-sig")
    (OUT / "monotonic_forward_summary.json").write_text(json.dumps({"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "rows": summary.to_dict(orient="records")}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
