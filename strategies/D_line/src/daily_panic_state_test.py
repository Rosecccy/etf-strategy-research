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


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "daily_panic_state_test"
DEV_YEARS = tuple(range(2016, 2024))
HOLDOUT_YEARS = (2024, 2025, 2026)
MARKET_THRESHOLDS = (40, 45, 50, 55, 60, 65)
SYMBOL_THRESHOLDS = (40, 50, 60, 70)
FORMAL_BASELINE = ROOT / "out" / "fear_greed" / "weight_sweep" / "summary.json"


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def daily_candidates(
    raw: pd.DataFrame,
    pool: pd.DataFrame,
    sentiment: pd.DataFrame,
) -> pd.DataFrame:
    symbol_fear = fg.build_symbol_fear(raw)
    market = sentiment[
        ["date", "fear_core", "fear_qvix", "greed_enhanced"]
    ].copy()
    market["fear_weighted"] = sweep.composite(
        market["fear_core"], market["fear_qvix"], 0.2
    )
    panel = raw.merge(symbol_fear, on=["date", "symbol"], how="left")
    panel = panel.merge(market, on="date", how="left")
    panel = panel.sort_values(["symbol", "date"])
    panel["entry_date"] = panel.groupby("symbol")["date"].shift(-1)
    panel["entry_close"] = panel.groupby("symbol")["close"].shift(-1)
    panel = panel[
        panel["entry_date"].notna()
        & panel["fear_weighted"].ge(min(MARKET_THRESHOLDS))
        & panel["symbol_fear"].ge(min(SYMBOL_THRESHOLDS))
    ].copy()
    name_map = pool.set_index("symbol")["name"].to_dict()
    panel["signal_date"] = panel["date"]
    panel["name"] = panel["symbol"].map(name_map).fillna("")
    panel["test_year"] = panel["entry_date"].dt.year
    panel["exit_date"] = pd.NaT
    panel["exit_close"] = np.nan
    panel["status"] = "open"
    panel["return_rate"] = np.nan
    panel["mark_close"] = panel.groupby("symbol")["close"].transform("last")
    panel["unrealized_return"] = panel["mark_close"] / panel["entry_close"] - 1.0
    panel["buy_quality"] = (
        0.5 * panel["fear_weighted"] + 0.5 * panel["symbol_fear"]
    )
    keep = [
        "test_year",
        "symbol",
        "name",
        "signal_date",
        "entry_date",
        "entry_close",
        "exit_date",
        "exit_close",
        "status",
        "return_rate",
        "mark_close",
        "unrealized_return",
        "fear_core",
        "fear_qvix",
        "fear_weighted",
        "greed_enhanced",
        "symbol_fear",
        "buy_quality",
    ]
    return panel[keep].reset_index(drop=True)


def main() -> None:
    configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
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
    rows: list[dict] = []
    logs: dict[tuple[int, int, str], pd.DataFrame] = {}

    for market_threshold in MARKET_THRESHOLDS:
        for symbol_threshold in SYMBOL_THRESHOLDS:
            selected = adjusted[
                adjusted["fear_weighted"].ge(market_threshold)
                & adjusted["symbol_fear"].ge(symbol_threshold)
            ].copy()
            for stage, years in (
                ("development", DEV_YEARS),
                ("holdout", HOLDOUT_YEARS),
            ):
                local = selected[selected["test_year"].isin(years)].copy()
                log, stat = portfolio_metrics(local, raw, pool)
                logs[(market_threshold, symbol_threshold, stage)] = log
                eligible = (
                    stat["closed_trades"] >= 10
                    if stage == "development"
                    else True
                )
                rows.append(
                    {
                        "market_threshold": market_threshold,
                        "symbol_threshold": symbol_threshold,
                        "stage": stage,
                        "eligible": eligible,
                        "score": score(stat) if eligible else -np.inf,
                        **stat,
                    }
                )

    result = pd.DataFrame(rows)
    dev = result[
        result["stage"].eq("development") & result["eligible"]
    ].sort_values(["score", "final_value", "win_rate"], ascending=False)
    chosen_market = int(dev.iloc[0]["market_threshold"])
    chosen_symbol = int(dev.iloc[0]["symbol_threshold"])
    held = result[
        result["stage"].eq("holdout")
        & result["market_threshold"].eq(chosen_market)
        & result["symbol_threshold"].eq(chosen_symbol)
    ].iloc[0]
    formal_summary = json.loads(FORMAL_BASELINE.read_text(encoding="utf-8"))
    baseline = formal_summary["winner_holdout_portfolio"]
    nearby = result[
        result["stage"].eq("holdout")
        & result["market_threshold"].between(chosen_market - 5, chosen_market + 5)
        & result["symbol_threshold"].between(chosen_symbol - 10, chosen_symbol + 10)
    ]
    passing = nearby[
        nearby["final_value"].gt(float(baseline["final_value"]) * 1.10)
        & nearby["win_rate"].ge(0.55)
        & nearby["max_drawdown"].ge(-0.15)
        & nearby["closed_trades"].ge(8)
    ]
    promoted = bool(
        held["final_value"] > float(baseline["final_value"]) * 1.10
        and held["win_rate"] >= 0.55
        and held["max_drawdown"] >= -0.15
        and held["closed_trades"] >= 8
        and len(passing) >= 4
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test": "D daily market-plus-symbol panic state candidates",
        "fixed_rules": {
            "execution": "signal after close, T+1 close entry and exit",
            "greed_exit": 70,
            "max_hold": 90,
            "portfolio": "one position, full allocation",
            "fees": "0.03%, minimum CNY 5 per side",
        },
        "development_years": "2016-2023",
        "holdout_years": "2024-2026",
        "development_selected": {
            "market_threshold": chosen_market,
            "symbol_threshold": chosen_symbol,
        },
        "formal_baseline_holdout": baseline,
        "winner_holdout": held.to_dict(),
        "nearby_variants_passing": int(len(passing)),
        "promoted": promoted,
    }
    result.to_csv(OUT / "variants.csv", index=False, encoding="utf-8-sig")
    logs[(chosen_market, chosen_symbol, "holdout")].to_csv(
        OUT / "winner_holdout_trades.csv", index=False, encoding="utf-8-sig"
    )
    passing.to_csv(OUT / "passing_neighbors.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
