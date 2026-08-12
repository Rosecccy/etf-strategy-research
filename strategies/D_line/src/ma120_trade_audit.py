from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

import factor_dca_scan as dca
import gate_nested_oos as gated
import nested_rule_oos as nested
import rolling_dca_validate as rolling
import stability_sweep as sweep


OUT = rolling.OUT
GATE = "self_ma120"


def build_trade_log(
    buy: dca.Candidate,
    sell: dca.Candidate,
    panel: pd.DataFrame,
    groups: dict,
    entry_start: pd.Timestamp,
    entry_end: pd.Timestamp,
    test_year: int,
    rule: str,
) -> pd.DataFrame:
    closes = panel["close"].to_numpy(dtype=float)
    dates = pd.to_datetime(panel["date"]).to_numpy()
    rows: list[dict] = []
    for start, end in groups["all"]:
        buy_signal = buy.mask[start:end]
        sell_signal = sell.mask[start:end]
        local_dates = pd.DatetimeIndex(dates[start:end])
        local_close = closes[start:end]
        entries = np.flatnonzero(buy_signal[:-1]) + 1
        entries = entries[(local_dates[entries] >= entry_start) & (local_dates[entries] < entry_end)]
        sell_exec = np.flatnonzero(sell_signal[:-1]) + 1
        if not len(entries):
            continue
        if len(sell_exec):
            position = np.searchsorted(sell_exec, entries + dca.MIN_HOLD_DAYS, side="left")
            valid = position < len(sell_exec)
        else:
            position = np.zeros(len(entries), dtype=int)
            valid = np.zeros(len(entries), dtype=bool)
        symbol = str(panel.iloc[start]["symbol"]).zfill(6)
        name = str(panel.iloc[start]["name"])
        for entry, has_exit, exit_position in zip(entries, valid, position):
            exit_ = sell_exec[exit_position] if has_exit else None
            return_rate = local_close[exit_] / local_close[entry] - 1 if has_exit else np.nan
            rows.append(
                {
                    "test_year": test_year,
                    "rule": rule,
                    "buy_combo": buy.label,
                    "sell_combo": sell.label,
                    "symbol": symbol,
                    "name": name,
                    "entry_date": local_dates[entry].date().isoformat(),
                    "exit_date": local_dates[exit_].date().isoformat() if has_exit else "",
                    "entry_close": local_close[entry],
                    "exit_close": local_close[exit_] if has_exit else np.nan,
                    "hold_days": int(exit_ - entry) if has_exit else np.nan,
                    "status": "closed" if has_exit else "open",
                    "return_rate": return_rate,
                    "pnl_cny": return_rate * 100 if has_exit else np.nan,
                }
            )
    return pd.DataFrame(rows)


def capacity(log: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    events: list[tuple[pd.Timestamp, int, str]] = []
    for _, trade in log.iterrows():
        events.append((pd.Timestamp(trade["entry_date"]), 1, str(trade["symbol"])))
        if trade["status"] == "closed":
            events.append((pd.Timestamp(trade["exit_date"]), -1, str(trade["symbol"])))
    # Conservative: buys at a close are counted before same-close sales.
    events.sort(key=lambda item: (item[0], 0 if item[1] > 0 else 1))
    total_open = 0
    symbol_open: dict[str, int] = {}
    max_total = 0
    max_symbol = 0
    rows: list[dict] = []
    for date, change, symbol in events:
        total_open += change
        symbol_open[symbol] = symbol_open.get(symbol, 0) + change
        max_total = max(max_total, total_open)
        max_symbol = max(max_symbol, symbol_open[symbol])
        rows.append(
            {
                "date": date.date().isoformat(),
                "symbol": symbol,
                "change": change,
                "open_batches_after_event": total_open,
                "symbol_open_batches_after_event": symbol_open[symbol],
            }
        )
    return pd.DataFrame(rows), max_total, max_symbol


def single_account_metrics(log: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float | str]:
    """Translate fixed 100-CNY lots into one no-deposit cash account."""
    events: list[tuple[pd.Timestamp, int, float]] = []
    for _, trade in log.iterrows():
        events.append((pd.Timestamp(trade["entry_date"]), 0, -100.0))
        if trade["status"] == "closed":
            events.append((pd.Timestamp(trade["exit_date"]), 1, 100.0 + float(trade["pnl_cny"])))
    # Conservative: same-close buys consume cash before same-close sells release it.
    events.sort(key=lambda item: (item[0], item[1]))
    cash_flow = 0.0
    minimum_cash_flow = 0.0
    for _, _, amount in events:
        cash_flow += amount
        minimum_cash_flow = min(minimum_cash_flow, cash_flow)
    initial_capital = -minimum_cash_flow
    open_lots = log[log["status"].eq("open")]
    open_value = float((open_lots["mark_close"] / open_lots["entry_close"] * 100).sum())
    final_value = initial_capital + cash_flow + open_value
    first_entry = pd.Timestamp(log["entry_date"].min())
    years = (as_of - first_entry).days / 365.25
    cagr = (final_value / initial_capital) ** (1 / years) - 1 if initial_capital > 0 and years > 0 else np.nan
    capital_days = 0.0
    for _, trade in log.iterrows():
        end = pd.Timestamp(trade["exit_date"]) if trade["status"] == "closed" else as_of
        capital_days += 100.0 * (end - pd.Timestamp(trade["entry_date"])).days
    average_deployed = capital_days / max((as_of - first_entry).days, 1)
    deployed_annual_return = (final_value - initial_capital) / (capital_days / 365.25) if capital_days else np.nan
    return {
        "first_entry_date": str(first_entry.date()),
        "as_of_date": str(as_of.date()),
        "years": float(years),
        "minimum_initial_capital_cny": float(initial_capital),
        "final_account_value_cny": float(final_value),
        "account_cagr": float(cagr),
        "time_weighted_average_deployed_cny": float(average_deployed),
        "annual_return_on_deployed_capital": float(deployed_annual_return),
    }


def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    full, groups = dca.load_panel()
    buy_conditions, buy_labels, buy_cores = dca.build_conditions(full, "buy")
    sell_conditions, sell_labels, sell_cores = dca.build_conditions(full, "sell")
    ma120_gate = gated.gates(full)[GATE]
    buy_conditions = {key: value & ma120_gate for key, value in buy_conditions.items()}
    max_year = min(rolling.LAST_TEST_YEAR, int(pd.to_datetime(full["date"]).dt.year.max()))

    train_cache: dict[int, list] = {}
    for year in range(rolling.FIRST_TEST_YEAR, max_year + 1):
        train, indices = rolling.training_panel(full, pd.Timestamp(year=year, month=1, day=1))
        if train.empty or train["symbol"].nunique() < 5:
            continue
        train_cache[year] = sweep.candidate_metrics_for_year(
            train,
            buy_conditions,
            sell_conditions,
            indices,
            buy_labels,
            buy_cores,
            sell_labels,
            sell_cores,
        )

    selections = {}
    for selector in sweep.RULES:
        for year, candidates in train_cache.items():
            chosen = sweep.choose(candidates, selector)
            if chosen is not None:
                selections[(selector.key, year)] = chosen

    validation_cache: dict = {}
    trade_parts: list[pd.DataFrame] = []
    selection_rows: list[dict] = []
    for year in range(nested.FIRST_NESTED_YEAR, max_year + 1):
        options = []
        for selector in sweep.RULES:
            metric = nested.validation_metric(selector, year, selections, full, groups, validation_cache)
            if metric is not None:
                options.append((selector, metric))
        if not options:
            continue
        selector, validation = max(
            options,
            key=lambda item: (
                item[1]["validation_win_rate"],
                item[1]["validation_avg_return"],
                item[1]["validation_closed_batches"],
            ),
        )
        buy, sell, train_metrics = selections[(selector.key, year)]
        trades = build_trade_log(
            buy,
            sell,
            full,
            groups,
            pd.Timestamp(year=year, month=1, day=1),
            pd.Timestamp(year=year + 1, month=1, day=1),
            year,
            selector.key,
        )
        trade_parts.append(trades)
        selection_rows.append(
            {
                "test_year": year,
                "rule": selector.key,
                "validation_closed_batches": validation["validation_closed_batches"],
                "validation_win_rate": validation["validation_win_rate"],
                "buy_id": buy.identifier,
                "buy_combo": buy.label,
                "sell_id": sell.identifier,
                "sell_combo": sell.label,
                "train_closed_batches": train_metrics["closed_batches"],
            }
        )

    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    last_close = full.groupby("symbol", sort=False)["close"].last().to_dict()
    trades["mark_close"] = trades["symbol"].map(last_close)
    trades["unrealized_return"] = np.where(
        trades["status"].eq("open"), trades["mark_close"] / trades["entry_close"] - 1, np.nan
    )
    trades["unrealized_pnl_cny"] = trades["unrealized_return"] * 100
    trades.to_csv(OUT / "ma120_trade_log.csv", index=False, encoding="utf-8-sig")
    selections_frame = pd.DataFrame(selection_rows)
    selections_frame.to_csv(OUT / "ma120_selected_rules.csv", index=False, encoding="utf-8-sig")
    closed = trades[trades["status"].eq("closed")].copy()
    open_lots = trades[trades["status"].eq("open")].copy()
    annual = (
        closed.groupby("test_year", as_index=False)
        .agg(
            completed_batches=("return_rate", "size"),
            win_batches=("return_rate", lambda values: int((values > 0).sum())),
            win_rate=("return_rate", lambda values: float((values > 0).mean())),
            avg_return=("return_rate", "mean"),
            total_pnl_cny=("pnl_cny", "sum"),
        )
        if not trades.empty
        else pd.DataFrame()
    )
    annual.to_csv(OUT / "ma120_annual_audit.csv", index=False, encoding="utf-8-sig")
    timeline, max_total, max_symbol = capacity(trades)
    timeline.to_csv(OUT / "ma120_capacity_timeline.csv", index=False, encoding="utf-8-sig")
    as_of = pd.to_datetime(full["date"]).max()
    account = single_account_metrics(trades, as_of)

    peak = timeline.loc[timeline["open_batches_after_event"].idxmax()] if not timeline.empty else pd.Series()
    name_lookup = full.drop_duplicates("symbol").set_index("symbol")["name"].to_dict()
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": "Nested rolling D strategy with self close >= self MA120 buy gate.",
        "test_years": f"{nested.FIRST_NESTED_YEAR}-{max_year}",
        "all_buy_batches": int(len(trades)),
        "open_batches_as_of_data_end": int(trades["status"].eq("open").sum()),
        "completed_batches": int(len(closed)),
        "win_batches": int((closed["return_rate"] > 0).sum()),
        "win_rate": float((closed["return_rate"] > 0).mean()),
        "average_return": float(closed["return_rate"].mean()),
        "realized_pnl_cny": float(closed["pnl_cny"].sum()),
        "unrealized_pnl_cny": float(open_lots["unrealized_pnl_cny"].sum()),
        "total_mark_to_market_pnl_cny": float(closed["pnl_cny"].sum() + open_lots["unrealized_pnl_cny"].sum()),
        "open_lot_cost_cny": float(len(open_lots) * 100),
        "open_lot_mark_value_cny": float(len(open_lots) * 100 + open_lots["unrealized_pnl_cny"].sum()),
        "max_concurrent_batches_conservative": int(max_total),
        "max_concurrent_notional_cny_conservative": int(max_total * 100),
        "max_concurrent_date": str(peak.get("date", "")),
        "max_concurrent_event_symbol": str(peak.get("symbol", "")),
        "max_concurrent_event_name": str(name_lookup.get(str(peak.get("symbol", "")).zfill(6), "")),
        "max_concurrent_batches_single_etf": int(max_symbol),
        "accounting": "Each lot is a fixed 100 CNY purchase. Profit is summed, not compounded; no fee or funding constraint is modeled.",
        **account,
    }
    (OUT / "ma120_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    shown = annual.copy()
    for column in ("win_rate", "avg_return"):
        shown[column] = shown[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "")
    report = [
        "# D 线 MA120 滚动策略：逐笔与资金容量审计",
        "",
        "策略为 D 线嵌套年度滚动选择器，加上‘自身收盘价不低于自身120日均线’的买入过滤。每笔低位信号在下一交易日收盘买入 100 元；卖点在下一交易日收盘卖出持有至少 7 个交易日的该批次。",
        "",
        f"- 全部买入批次：{summary['all_buy_batches']}；其中已完成 {summary['completed_batches']} 批，数据截止日仍持有 {summary['open_batches_as_of_data_end']} 批。",
        f"- 已完成批次中盈利 {summary['win_batches']} 批；胜率：{summary['win_rate']:.2%}。",
        f"- 平均单批收益：{summary['average_return']:.2%}；已实现利润合计：{summary['realized_pnl_cny']:.2f} 元。",
        f"- 未平仓批次成本 {summary['open_lot_cost_cny']:.2f} 元，按各 ETF 最新可用收盘价浮动盈亏 {summary['unrealized_pnl_cny']:.2f} 元，市值 {summary['open_lot_mark_value_cny']:.2f} 元。",
        f"- 已实现加浮动的合计盈亏：{summary['total_mark_to_market_pnl_cny']:.2f} 元。",
        f"- 最大同时持有：{summary['max_concurrent_batches_conservative']} 批，即按每批 100 元计算需准备 {summary['max_concurrent_notional_cny_conservative']} 元，发生于 {summary['max_concurrent_date']}。",
        f"- 单一 ETF 最大同时持有：{summary['max_concurrent_batches_single_etf']} 批。",
        "- 容量采用保守口径：同日买入先于同日卖出计入，因此不会低估当日所需资金。",
        "",
        "## 单账户折算",
        f"- 若从 {summary['first_entry_date']} 一次性准备最小本金 {summary['minimum_initial_capital_cny']:.2f} 元，期间不追加资金，至 {summary['as_of_date']} 的账户估值为 {summary['final_account_value_cny']:.2f} 元。",
        f"- 对应年化复合收益率（CAGR）：{summary['account_cagr']:.2%}。",
        f"- 策略平均实际占用资金为 {summary['time_weighted_average_deployed_cny']:.2f} 元；按占用资金口径的年度利润率为 {summary['annual_return_on_deployed_capital']:.2%}。",
        "",
        "## 年度已平仓结果",
        shown.to_markdown(index=False),
    ]
    (OUT / "ma120_audit_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
