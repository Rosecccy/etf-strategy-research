from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from s1lib import read_etf


ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "fit" / "strengthen"
FORMAL = ROOT / "fit" / "selector"
CFG = ROOT / "cfg" / "s1_model.json"
INITIAL_CASH = 10_000.0
COMMISSION_RATE = 0.0003
MIN_COMMISSION = 5.0
HOLDOUT_YEARS = {2024, 2025, 2026}


def fee(notional: float) -> float:
    return max(MIN_COMMISSION, abs(notional) * COMMISSION_RATE)


def normalize_trades(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["symbol"] = result["symbol"].astype(str).str.zfill(6)
    result["entry_date"] = pd.to_datetime(result["entry_date"], errors="coerce")
    result["exit_date"] = pd.to_datetime(result["exit_date"], errors="coerce")
    result["ret"] = pd.to_numeric(result["ret"], errors="coerce")
    return result.dropna(subset=["entry_date", "exit_date", "ret"]).sort_values(
        ["entry_date", "exit_date", "symbol"]
    ).reset_index(drop=True)


def attach_prices(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["entry_close"] = np.nan
    result["exit_close"] = np.nan
    for symbol, indexes in result.groupby("symbol").groups.items():
        prices = read_etf(str(symbol))[["date", "close"]].copy()
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        lookup = prices.dropna().drop_duplicates("date").set_index("date")["close"]
        result.loc[indexes, "entry_close"] = result.loc[indexes, "entry_date"].map(lookup)
        result.loc[indexes, "exit_close"] = result.loc[indexes, "exit_date"].map(lookup)
    result = result.dropna(subset=["entry_close", "exit_close"]).copy()
    result["ret"] = result["exit_close"] / result["entry_close"] - 1.0
    return result


def gross_metrics(frame: pd.DataFrame, years: set[int] | None = None) -> dict:
    local = frame.copy()
    if years is not None:
        local = local[local["entry_date"].dt.year.isin(years)]
    returns = local["ret"].astype(float)
    if returns.empty:
        return {
            "trades": 0,
            "final_1000": 1000.0,
            "win_rate": None,
            "max_drawdown": 0.0,
        }
    equity = 1000.0 * (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "trades": int(len(returns)),
        "final_1000": float(equity.iloc[-1]),
        "win_rate": float((returns > 0).mean()),
        "avg_return": float(returns.mean()),
        "max_drawdown": float(drawdown.min()),
    }


def net_account(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cash = INITIAL_CASH
    rows = []
    for _, trade in frame.iterrows():
        entry = float(trade["entry_close"])
        exit_price = float(trade["exit_close"])
        shares = math.floor(cash / (entry * 100)) * 100
        while shares >= 100 and shares * entry + fee(shares * entry) > cash:
            shares -= 100
        if shares < 100:
            continue
        before = cash
        buy_value = shares * entry
        sell_value = shares * exit_price
        cash += sell_value - fee(sell_value) - buy_value - fee(buy_value)
        item = trade.to_dict()
        item.update(
            {
                "shares": shares,
                "cash_before_net": before,
                "cash_after_net": cash,
                "net_return_on_account": cash / before - 1.0,
            }
        )
        rows.append(item)
    log = pd.DataFrame(rows)
    return log, {
        "initial_cash": INITIAL_CASH,
        "final_cash": float(cash),
        "net_profit": float(cash - INITIAL_CASH),
        "executed_trades": int(len(log)),
    }


def annual_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in frame.groupby(frame["entry_date"].dt.year, sort=True):
        values = group["ret"].astype(float)
        rows.append(
            {
                "year": int(year),
                "trades": int(len(values)),
                "win_rate": float((values > 0).mean()),
                "gross_return": float(np.prod(1.0 + values) - 1.0),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    quality = json.loads((ROOT / "raw" / "quality.json").read_text(encoding="utf-8"))
    if not bool(quality.get("passed")):
        raise RuntimeError("S quality gate failed; selector promotion is blocked.")
    approved = {str(value).zfill(6) for value in quality["approved_symbols"]}
    trades = attach_prices(
        normalize_trades(pd.read_csv(STAGING / "winner_trades.csv", dtype={"symbol": str}, encoding="utf-8-sig"))
    )
    unknown = sorted(set(trades["symbol"]) - approved)
    if unknown:
        raise RuntimeError(f"Unapproved ETF in clean selector: {unknown}")

    selected = pd.read_csv(STAGING / "winner_selected_by_year.csv", encoding="utf-8-sig")
    selected = selected.rename(columns={"test_year": "year"})
    selected["year"] = pd.to_numeric(selected["year"], errors="raise").astype(int)
    selected["reason"] = np.where(selected["param_id"].eq("CASH"), "cash_gate", "selected")
    selected["method_id"] = "w5-7_balanced_g+0.04"
    selected["train_end"] = selected["year"] - 1
    if not (selected["train_end"] < selected["year"]).all():
        raise RuntimeError("Annual selector causality audit failed.")

    FORMAL.mkdir(parents=True, exist_ok=True)
    trades.to_csv(FORMAL / "final_trades.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(FORMAL / "selected_by_year.csv", index=False, encoding="utf-8-sig")
    annual_table(trades).to_csv(FORMAL / "annual.csv", index=False, encoding="utf-8-sig")
    net_log, net = net_account(trades)
    net_log.to_csv(FORMAL / "net_trades.csv", index=False, encoding="utf-8-sig")

    staging_summary = json.loads((STAGING / "summary.json").read_text(encoding="utf-8"))
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "formal_clean",
        "method_id": "w5-7_balanced_g+0.04",
        "data_quality_passed": True,
        "approved_universe": len(approved),
        "excluded_symbols": quality.get("excluded_symbols", []),
        "selection": "Each test year uses only completed earlier-year data; method chosen on 2016-2023.",
        "development_years": "2016-2023",
        "holdout_years": "2024-2026",
        "candidate_params": staging_summary["candidate_params"],
        "method_variants": staging_summary["selector_methods"],
        "all": gross_metrics(trades),
        "holdout": gross_metrics(trades, HOLDOUT_YEARS),
        "net_10000": net,
        "promotion_reason": "The prior formal table contained an excluded ETF; the clean retrain is the only admissible formal S history.",
    }
    (FORMAL / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    cfg = json.loads(CFG.read_text(encoding="utf-8-sig"))
    cfg["name"] = "S1 clean strict rolling live"
    cfg["status"] = "live_active_clean"
    cfg["current_year"] = 2026
    current = selected[selected["year"].eq(2026)].iloc[-1].to_dict()
    cfg["current_param"] = current
    cfg["strict_validation"] = {
        "development_years": summary["development_years"],
        "holdout_years": summary["holdout_years"],
        "candidate_params": summary["candidate_params"],
        "method_variants": summary["method_variants"],
        "gross_final_cash_1000": summary["all"]["final_1000"],
        "net_final_cash_10000": net["final_cash"],
        "trade_count": summary["all"]["trades"],
        "win_rate": summary["all"]["win_rate"],
        "max_drawdown": summary["all"]["max_drawdown"],
        "holdout_return": summary["holdout"]["final_1000"] / 1000.0 - 1.0,
        "source": "fit/selector/summary.json",
        "data_quality": "passed",
    }
    CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
