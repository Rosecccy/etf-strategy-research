from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "D" / "out" / "csd_v2_live_month_review"
TRADE_PATH = ROOT / "D" / "out" / "csd_v2_exit_guard" / "strict_rolling_conservative_tp_center15_profit_first_trades.csv"
DAILY_PATH = ROOT / "D" / "out" / "line_specific_gates" / "rolling_combo_daily.csv"
C_RAW = ROOT / "C" / "raw" / "etf"
S_RAW = ROOT / "S" / "raw" / "etf"


def load_price(symbol: str) -> pd.DataFrame:
    for root in (C_RAW, S_RAW):
        p = root / f"{symbol}.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"], dtype={"symbol": str})
            return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    raise FileNotFoundError(symbol)


def close_on_or_before(symbol: str, date: pd.Timestamp) -> tuple[pd.Timestamp, float]:
    px = load_price(symbol)
    part = px[px["date"].le(date)]
    if part.empty:
        raise ValueError(f"No price before {date} for {symbol}")
    row = part.iloc[-1]
    return pd.Timestamp(row["date"]), float(row["close"])


def add_prices(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for row in df.itertuples(index=False):
        entry_price_date, entry_close = close_on_or_before(str(row.symbol), row.entry_date)
        actual_exit = pd.Timestamp(row.exit_date_new)
        status = "已完成" if actual_exit <= cutoff else "未完成_按最新价暂估"
        settle_date = min(actual_exit, cutoff)
        exit_price_date, exit_close = close_on_or_before(str(row.symbol), settle_date)
        ret = exit_close / entry_close - 1.0
        rows.append(
            {
                "signal_date": row.signal_date,
                "line": row.line,
                "symbol": row.symbol,
                "name": row.name,
                "entry_date": row.entry_date,
                "entry_price_date": entry_price_date,
                "entry_close": entry_close,
                "planned_exit_date": row.exit_date_new,
                "settle_date": exit_price_date,
                "exit_close": exit_close,
                "status": status,
                "exit_reason": row.exit_reason,
                "ret": ret,
                "profit_if_100": ret * 100.0,
                "profit_if_1000": ret * 1000.0,
                "policy": row.policy,
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "trades": 0,
            "completed": 0,
            "open": 0,
            "win_rate": 0.0,
            "avg_ret": 0.0,
            "sum_ret": 0.0,
            "profit_if_100_each": 0.0,
            "profit_if_1000_each": 0.0,
        }
    r = df["ret"].astype(float)
    return {
        "trades": int(len(df)),
        "completed": int(df["status"].eq("已完成").sum()),
        "open": int(df["status"].ne("已完成").sum()),
        "win_rate": float((r > 0).mean()),
        "avg_ret": float(r.mean()),
        "sum_ret": float(r.sum()),
        "profit_if_100_each": float(df["profit_if_100"].sum()),
        "profit_if_1000_each": float(df["profit_if_1000"].sum()),
    }


def window_report(trades: pd.DataFrame, daily: pd.DataFrame, start: str, end: str, label: str, cutoff: str) -> tuple[pd.DataFrame, dict]:
    start_ts, end_ts, cutoff_ts = pd.Timestamp(start), pd.Timestamp(end), pd.Timestamp(cutoff)
    # Real account activity inside the window: positions that overlap the window.
    overlap = trades[(trades["entry_date"].le(end_ts)) & (trades["exit_date_new"].ge(start_ts))].copy()
    detail = add_prices(overlap, cutoff_ts) if len(overlap) else pd.DataFrame()
    if len(detail):
        detail = detail.sort_values(["settle_date", "entry_date", "symbol"]).reset_index(drop=True)
        detail.to_csv(OUT / f"{label}_trade_detail.csv", index=False, encoding="utf-8-sig")

    # Daily line status tells whether there were new official entries.
    d = daily[(daily["date"].between(start_ts, end_ts))].copy()
    active_days = int(d["active"].eq(True).sum()) if len(d) else 0
    summary = summarize(detail)
    summary.update(
        {
            "label": label,
            "window_start": start,
            "window_end": end,
            "cutoff": cutoff,
            "official_new_entry_days": active_days,
            "cash_days": int(d["active"].eq(False).sum()) if len(d) else 0,
            "daily_rows": int(len(d)),
        }
    )
    return detail, summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trades = pd.read_csv(
        TRADE_PATH,
        parse_dates=["signal_date", "entry_date", "exit_date_new", "original_exit_date"],
        dtype={"symbol": str},
    )
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    latest_data_date = max(
        pd.read_csv(p, usecols=["date"])["date"].max()
        for p in list(C_RAW.glob("*.csv")) + list(S_RAW.glob("*.csv"))
    )
    cutoff = min(pd.Timestamp("2026-08-10"), pd.Timestamp(latest_data_date)).strftime("%Y-%m-%d")

    windows = [
        ("rolling_last_month", "2026-07-10", cutoff),
        ("calendar_2026_07", "2026-07-01", "2026-07-31"),
    ]
    summaries = []
    details = {}
    for label, start, end in windows:
        detail, summary = window_report(trades, daily, start, end, label, cutoff)
        summaries.append(summary)
        details[label] = detail

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# C/S/D v2 + 15%止盈治理：近一个月实盘化复盘",
        "",
        f"本地行情最新日期：{cutoff}。今日 2026-08-10 尚无本地 8月10日收盘数据，所以未完成交易按 {cutoff} 最新价暂估。",
        "",
        "策略口径：使用 `conservative_tp_center15_profit_first`，也就是 C/S/D v2 买点与裁判器不变，只叠加严格滚动验证过的 15% 附近止盈治理。",
        "",
        "重要说明：本复盘按“所有独立交易”统计，不是单仓账户资金曲线；若每笔固定买 100 元，则盈利等于收益率之和乘以 100。",
        "",
    ]
    for summary in summaries:
        lines.extend(
            [
                f"## {summary['label']}",
                f"- 窗口：{summary['window_start']} 至 {summary['window_end']}",
                f"- 官方新开仓天数：{summary['official_new_entry_days']}",
                f"- 空仓天数：{summary['cash_days']} / {summary['daily_rows']}",
                f"- 与窗口有交集的持仓/交易：{summary['trades']} 笔",
                f"- 已完成：{summary['completed']} 笔；未完成按最新价暂估：{summary['open']} 笔",
                f"- 胜率：{summary['win_rate']:.2%}",
                f"- 平均单笔：{summary['avg_ret']:.2%}",
                f"- 收益率之和：{summary['sum_ret']:.2%}",
                f"- 若每笔买 100 元：{summary['profit_if_100_each']:.2f} 元",
                f"- 若每笔买 1000 元：{summary['profit_if_1000_each']:.2f} 元",
                "",
            ]
        )
        detail = details[summary["label"]]
        if len(detail):
            lines.append("| 线 | ETF | 买入日 | 结算/卖出日 | 状态 | 收益 | 每100元盈亏 | 原因 |")
            lines.append("|---|---|---|---|---|---:|---:|---|")
            for r in detail.itertuples(index=False):
                lines.append(
                    f"| {r.line} | {r.symbol} {r.name} | {pd.Timestamp(r.entry_date).date()} | {pd.Timestamp(r.settle_date).date()} | {r.status} | {r.ret:.2%} | {r.profit_if_100:.2f} | {r.exit_reason} |"
                )
            lines.append("")
    (OUT / "复盘说明.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary_df.to_string(index=False))
    print(OUT)


if __name__ == "__main__":
    main()
