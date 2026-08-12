from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import BDay

import fear_greed_oos as fg
import fear_greed_weight_sweep as sweep
from control_panic_age_test import portfolio_metrics
from daily_panic_rolling_test import prepare
from daily_panic_stale_exit_test import Rule, adjust_exits


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "cfg" / "daily_panic.json"
LIVE = ROOT / "live"


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def selected_rule(config: dict, year: int) -> Rule | None:
    table = config.get("stale_recovery_exit", {}).get("selected_by_year", {})
    values = table.get(str(year))
    if not values:
        return None
    return Rule(
        int(values["min_hold"]),
        float(values["peak_cap"]),
        float(values["recovery_target"]),
    )


def build_live_candidates(adjusted: pd.DataFrame, raw: pd.DataFrame, config: dict) -> pd.DataFrame:
    market_threshold = float(config["market_fear_threshold"])
    symbol_threshold = float(config["symbol_fear_threshold"])
    formal = adjusted[
        adjusted["test_year"].ge(2024)
        & adjusted["fear_weighted"].ge(market_threshold)
        & adjusted["symbol_fear"].ge(symbol_threshold)
    ].copy()
    parts = []
    for year in range(2024, int(raw["date"].max().year) + 1):
        rule = selected_rule(config, year)
        source = formal if rule is None else adjust_exits(formal, raw, rule)
        parts.append(source[source["test_year"].eq(year)].copy())
    return pd.concat(parts, ignore_index=True, sort=False) if parts else formal.iloc[0:0].copy()


def latest_scan(
    raw: pd.DataFrame,
    pool: pd.DataFrame,
    market_threshold: float,
    symbol_threshold: float,
    qvix_weight: float,
) -> tuple[pd.DataFrame, dict]:
    sentiment = fg.build_sentiment(raw)
    symbol_fear = fg.build_symbol_fear(raw)
    latest_market = sentiment.dropna(subset=["fear_core"]).iloc[-1]
    fear_score = float(
        sweep.composite(
            pd.Series([latest_market["fear_core"]]),
            pd.Series([latest_market["fear_qvix"]]),
            qvix_weight,
        )[0]
    )
    latest_date = pd.Timestamp(latest_market["date"])
    symbols = (
        symbol_fear[symbol_fear["date"].eq(latest_date)]
        .merge(
            raw[raw["date"].eq(latest_date)][["symbol", "close", "amount"]],
            on="symbol",
            how="left",
        )
        .merge(pool[["symbol", "name"]], on="symbol", how="left")
    )
    symbols["market_fear"] = fear_score
    symbols["buy_quality"] = 0.5 * symbols["market_fear"] + 0.5 * symbols["symbol_fear"]
    symbols["market_pass"] = fear_score >= market_threshold
    symbols["symbol_pass"] = symbols["symbol_fear"] >= symbol_threshold
    symbols["buy_pass"] = symbols["market_pass"] & symbols["symbol_pass"]
    symbols = symbols.sort_values(
        ["buy_pass", "buy_quality", "amount", "symbol"],
        ascending=[False, False, False, True],
    )
    return symbols, {
        "data_date": str(latest_date.date()),
        "market_fear": fear_score,
        "market_greed": 100.0 - fear_score,
        "market_pass": bool(fear_score >= market_threshold),
    }


def open_position_signal(position: pd.Series, raw: pd.DataFrame, market: dict, rule: Rule | None) -> dict:
    symbol = str(position["symbol"]).zfill(6)
    entry_date = pd.Timestamp(position["entry_date"])
    history = raw[(raw["symbol"].eq(symbol)) & raw["date"].ge(entry_date)].sort_values("date")
    entry_price = float(position["entry_close"])
    held = max(0, len(history) - 1)
    returns = history["close"].astype(float) / entry_price - 1.0
    peak_return = float(returns.max()) if len(returns) else 0.0
    current_return = float(returns.iloc[-1]) if len(returns) else 0.0
    reasons = []
    if held >= 90:
        reasons.append("达到90个交易日最长持有期")
    if held >= 7 and float(market["market_greed"]) >= 70:
        reasons.append("市场贪婪分达到70")
    stale = False
    if rule is not None:
        stale = bool(
            held >= rule.min_hold
            and peak_return < rule.peak_cap
            and current_return >= rule.recovery_target
        )
        if stale:
            reasons.append(
                f"滞涨仓位持有{held}天后恢复到{current_return:.2%}，触发回本退出"
            )
    return {
        "sell": bool(reasons),
        "reason": "；".join(reasons) if reasons else "尚未触发贪婪、最长持有或滞涨回本卖点",
        "held_trading_days": held,
        "peak_return": peak_return,
        "current_return": current_return,
        "stale_exit_triggered": stale,
        "active_stale_rule": rule.key if rule else "BASE_EXIT",
    }


def main() -> None:
    configure_stdout()
    LIVE.mkdir(parents=True, exist_ok=True)
    config = json.loads(CFG.read_text(encoding="utf-8"))
    market_threshold = float(config["market_fear_threshold"])
    symbol_threshold = float(config["symbol_fear_threshold"])
    qvix_weight = float(config["qvix_weight"])

    adjusted, raw, pool = prepare()
    formal = build_live_candidates(adjusted, raw, config)
    portfolio_log, portfolio = portfolio_metrics(formal, raw, pool)
    open_rows = portfolio_log[portfolio_log["status_portfolio"].eq("open_marked")]
    scan, market = latest_scan(raw, pool, market_threshold, symbol_threshold, qvix_weight)
    candidates = scan[scan["buy_pass"]].copy()
    data_date = pd.Timestamp(market["data_date"])
    execution_date = str((data_date + BDay(1)).date())
    current_rule = selected_rule(config, int(data_date.year))

    if not open_rows.empty:
        position = open_rows.iloc[-1]
        position_signal = open_position_signal(position, raw, market, current_rule)
        model_action = "卖出" if position_signal["sell"] else "继续持有"
        model_symbol = str(position["symbol"]).zfill(6)
        model_name = str(position["name"])
        model_reason = position_signal["reason"]
    else:
        position_signal = None
        model_symbol = ""
        model_name = ""
        if candidates.empty:
            model_action = "空仓等待"
            model_reason = "模型账户无持仓，今日也没有达到双恐慌阈值的候选"
        else:
            top = candidates.iloc[0]
            model_action = "买入"
            model_symbol = str(top["symbol"]).zfill(6)
            model_name = str(top["name"])
            model_reason = "模型账户空仓，最高分候选达到市场与ETF双恐慌阈值"

    if candidates.empty:
        cash_action = "空仓等待"
        cash_symbol = ""
        cash_name = ""
        cash_reason = (
            f"市场恐慌分{market['market_fear']:.2f}，没有ETF同时达到市场55和自身70的门槛"
        )
    else:
        top = candidates.iloc[0]
        cash_action = "下一交易日收盘买入"
        cash_symbol = str(top["symbol"]).zfill(6)
        cash_name = str(top["name"])
        cash_reason = (
            f"市场恐慌分{market['market_fear']:.2f}，ETF自身恐慌分{float(top['symbol_fear']):.2f}"
        )

    decision = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": config["version"],
        **market,
        "expected_execution_date": execution_date,
        "model_account": {
            "action": model_action,
            "symbol": model_symbol,
            "name": model_name,
            "reason": model_reason,
            "position_state": position_signal,
            "estimated_value": float(portfolio["final_value"]),
        },
        "if_real_account_is_cash": {
            "action": cash_action,
            "symbol": cash_symbol,
            "name": cash_name,
            "reason": cash_reason,
        },
        "candidate_count": int(len(candidates)),
    }
    (LIVE / "daily_panic_today.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "data_date": market["data_date"],
                "execution_date": execution_date,
                "model_action": model_action,
                "model_symbol": model_symbol,
                "model_name": model_name,
                "cash_action": cash_action,
                "cash_symbol": cash_symbol,
                "cash_name": cash_name,
                "market_fear": market["market_fear"],
                "candidate_count": len(candidates),
            }
        ]
    ).to_csv(LIVE / "daily_panic_today.csv", index=False, encoding="utf-8-sig")
    scan.to_csv(LIVE / "daily_panic_scan.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
