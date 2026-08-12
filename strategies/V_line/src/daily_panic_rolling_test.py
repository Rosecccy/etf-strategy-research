from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import fear_greed_oos as fg
import fear_greed_weight_sweep as sweep
from control_panic_age_test import portfolio_metrics, score
from daily_panic_state_test import (
    FORMAL_BASELINE,
    MARKET_THRESHOLDS,
    SYMBOL_THRESHOLDS,
    daily_candidates,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "daily_panic_rolling_test"
DEV_TEST_YEARS = tuple(range(2019, 2024))
HOLDOUT_YEARS = (2024, 2025, 2026)
WINDOWS: tuple[int | None, ...] = (3, 5, None)


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw, pool = fg.load_clean_raw()
    sentiment = fg.build_sentiment(raw)
    candidates = daily_candidates(raw, pool, sentiment)
    exit_sentiment = sentiment.copy()
    exit_sentiment["fear_enhanced"] = sweep.composite(
        exit_sentiment["fear_core"], exit_sentiment["fear_qvix"], 0.2
    )
    exit_sentiment["greed_enhanced"] = 100.0 - exit_sentiment["fear_enhanced"]
    adjusted = fg.apply_exit_rule(
        candidates,
        fg.ExitRule("daily_panic_h90_g70", 90, 70.0),
        raw,
        exit_sentiment,
    )
    return adjusted, raw, pool


def train_years(year: int, window: int | None) -> tuple[int, ...]:
    start = 2016 if window is None else max(2016, year - window)
    return tuple(range(start, year))


def choose_threshold(
    adjusted: pd.DataFrame,
    raw: pd.DataFrame,
    pool: pd.DataFrame,
    year: int,
    window: int | None,
) -> tuple[int, int, dict]:
    years = train_years(year, window)
    cutoff = pd.Timestamp(year=year, month=1, day=1)
    history = adjusted[
        adjusted["test_year"].isin(years)
        & adjusted["status"].eq("closed")
        & adjusted["exit_date"].lt(cutoff)
    ].copy()
    minimum = max(6, len(years) * 2)
    rows: list[dict] = []
    for market in MARKET_THRESHOLDS:
        for symbol in SYMBOL_THRESHOLDS:
            local = history[
                history["fear_weighted"].ge(market)
                & history["symbol_fear"].ge(symbol)
            ].copy()
            _, stat = portfolio_metrics(local, raw, pool)
            rows.append(
                {
                    "market_threshold": market,
                    "symbol_threshold": symbol,
                    "eligible": stat["closed_trades"] >= minimum,
                    "selection_score": (
                        score(stat) if stat["closed_trades"] >= minimum else -np.inf
                    ),
                    **stat,
                }
            )
    table = pd.DataFrame(rows).sort_values(
        ["selection_score", "final_value", "win_rate"],
        ascending=False,
    )
    eligible = table[np.isfinite(table["selection_score"])]
    if eligible.empty:
        return 55, 70, {"fallback": True, "training_trades": 0}
    winner = eligible.iloc[0]
    return (
        int(winner["market_threshold"]),
        int(winner["symbol_threshold"]),
        winner.to_dict(),
    )


def rolling_candidates(
    adjusted: pd.DataFrame,
    raw: pd.DataFrame,
    pool: pd.DataFrame,
    window: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    choices: list[dict] = []
    for year in range(2019, 2027):
        market, symbol, audit = choose_threshold(
            adjusted, raw, pool, year, window
        )
        current = adjusted[
            adjusted["test_year"].eq(year)
            & adjusted["fear_weighted"].ge(market)
            & adjusted["symbol_fear"].ge(symbol)
        ].copy()
        current["selected_market_threshold"] = market
        current["selected_symbol_threshold"] = symbol
        current["selector_window"] = "all" if window is None else window
        parts.append(current)
        choices.append(
            {
                "test_year": year,
                "window": "all" if window is None else window,
                "train_years": ",".join(map(str, train_years(year, window))),
                "market_threshold": market,
                "symbol_threshold": symbol,
                "train_score": audit.get("selection_score"),
                "train_closed_trades": audit.get("closed_trades"),
                "fallback": audit.get("fallback", False),
            }
        )
        print(
            f"[rolling {'all' if window is None else window}] "
            f"{year}: market={market}, symbol={symbol}"
        )
    return pd.concat(parts, ignore_index=True), pd.DataFrame(choices)


def main() -> None:
    configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    adjusted, raw, pool = prepare()
    rows: list[dict] = []
    generated: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for window in WINDOWS:
        key = "all" if window is None else str(window)
        candidates, choices = rolling_candidates(adjusted, raw, pool, window)
        generated[key] = (candidates, choices)
        for stage, years in (
            ("development", DEV_TEST_YEARS),
            ("holdout", HOLDOUT_YEARS),
        ):
            local = candidates[candidates["test_year"].isin(years)].copy()
            _, stat = portfolio_metrics(local, raw, pool)
            rows.append(
                {
                    "window": key,
                    "stage": stage,
                    "score": score(stat),
                    **stat,
                }
            )

    result = pd.DataFrame(rows)
    dev = result[result["stage"].eq("development")].sort_values(
        ["score", "final_value", "win_rate"], ascending=False
    )
    chosen = str(dev.iloc[0]["window"])
    candidates, choices = generated[chosen]
    held = result[
        result["stage"].eq("holdout") & result["window"].eq(chosen)
    ].iloc[0]
    formal = json.loads(FORMAL_BASELINE.read_text(encoding="utf-8"))[
        "winner_holdout_portfolio"
    ]
    passing = result[
        result["stage"].eq("holdout")
        & result["final_value"].gt(float(formal["final_value"]) * 1.10)
        & result["win_rate"].ge(0.55)
        & result["max_drawdown"].ge(-0.15)
        & result["closed_trades"].ge(8)
    ]
    promoted = bool(
        held["final_value"] > float(formal["final_value"]) * 1.10
        and held["win_rate"] >= 0.55
        and held["max_drawdown"] >= -0.15
        and held["closed_trades"] >= 8
        and len(passing) >= 2
    )
    holdout_candidates = candidates[
        candidates["test_year"].isin(HOLDOUT_YEARS)
    ].copy()
    holdout_log, _ = portfolio_metrics(holdout_candidates, raw, pool)
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test": "D annual past-only daily panic threshold selector",
        "development_selector_years": "2019-2023",
        "holdout_years": "2024-2026",
        "development_selected_window": chosen,
        "formal_baseline_holdout": formal,
        "winner_holdout": held.to_dict(),
        "rolling_windows_passing": int(len(passing)),
        "promoted": promoted,
    }
    result.to_csv(OUT / "window_results.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(OUT / "winner_choices.csv", index=False, encoding="utf-8-sig")
    holdout_log.to_csv(
        OUT / "winner_holdout_trades.csv", index=False, encoding="utf-8-sig"
    )
    passing.to_csv(OUT / "passing_windows.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
