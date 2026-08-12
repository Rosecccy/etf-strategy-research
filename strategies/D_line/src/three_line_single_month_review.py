from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "D" / "out" / "one_year_three_line_eval" / "trades.csv"
OUT = ROOT / "D" / "out" / "three_line_single_month_review"
C_RAW = ROOT / "C" / "raw" / "etf"
S_RAW = ROOT / "S" / "raw" / "etf"


def price(symbol: str) -> pd.DataFrame:
    for root in (C_RAW, S_RAW):
        p = root / f"{symbol}.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"], dtype={"symbol": str})
            return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    raise FileNotFoundError(symbol)


def close_on_or_before(symbol: str, date: pd.Timestamp) -> tuple[pd.Timestamp, float]:
    df = price(symbol)
    part = df[df["date"].le(date)]
    if part.empty:
        raise ValueError(f"No price for {symbol} before {date}")
    row = part.iloc[-1]
    return pd.Timestamp(row["date"]), float(row["close"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp("2026-07-10")
    cutoff = pd.Timestamp("2026-08-07")
    trades = pd.read_csv(SRC, parse_dates=["signal_date", "entry_date", "exit_date"], dtype={"symbol": str})
    # T+1 entry must have happened by cutoff. 2026-08-07 signals require 2026-08-10 execution, so exclude them here.
    sample = trades[
        trades["signal_date"].between(start, cutoff)
        & trades["entry_date"].notna()
        & trades["entry_date"].le(cutoff)
    ].copy()

    rows = []
    for r in sample.itertuples(index=False):
        entry_price_date, entry_close = close_on_or_before(str(r.symbol), r.entry_date)
        if bool(r.completed) and pd.notna(r.exit_date) and pd.Timestamp(r.exit_date) <= cutoff:
            settle_target = pd.Timestamp(r.exit_date)
            status = "已完成"
        else:
            settle_target = cutoff
            status = "未完成_按最新价暂估"
        settle_date, settle_close = close_on_or_before(str(r.symbol), settle_target)
        ret = settle_close / entry_close - 1.0
        rows.append(
            {
                "signal_date": r.signal_date,
                "line": r.line,
                "symbol": r.symbol,
                "name": r.name,
                "entry_date": r.entry_date,
                "entry_price_date": entry_price_date,
                "entry_close": entry_close,
                "planned_exit_date": r.exit_date,
                "settle_date": settle_date,
                "settle_close": settle_close,
                "status": status,
                "ret": ret,
                "profit_if_100": ret * 100.0,
                "profit_if_1000": ret * 1000.0,
            }
        )
    detail = pd.DataFrame(rows).sort_values(["line", "entry_date", "symbol"]).reset_index(drop=True)
    detail.to_csv(OUT / "single_line_month_detail.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for line, g in detail.groupby("line"):
        rr = g["ret"].astype(float)
        summary_rows.append(
            {
                "line": line,
                "trades": len(g),
                "completed": int(g["status"].eq("已完成").sum()),
                "open_mark_to_market": int(g["status"].ne("已完成").sum()),
                "win_rate": float((rr > 0).mean()) if len(rr) else 0.0,
                "avg_ret": float(rr.mean()) if len(rr) else 0.0,
                "sum_ret": float(rr.sum()) if len(rr) else 0.0,
                "profit_if_100_each": float((rr * 100).sum()) if len(rr) else 0.0,
                "profit_if_1000_each": float((rr * 1000).sum()) if len(rr) else 0.0,
                "best_trade": float(rr.max()) if len(rr) else 0.0,
                "worst_trade": float(rr.min()) if len(rr) else 0.0,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("sum_ret", ascending=False)
    summary.to_csv(OUT / "single_line_month_summary.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# C/S/D 三条线单独执行：近一个月复盘",
        "",
        "口径：使用 `D/out/one_year_three_line_eval/trades.csv`，也就是 C、S、D 三条线各自单独每日执行，不经过 C/S/D 合成裁判器过滤。",
        "",
        f"窗口：{start.date()} 至 {cutoff.date()}。本地行情最新到 {cutoff.date()}。",
        "",
        "统计方式：信号日位于窗口内，且 T+1 买入日已经发生；已完成交易按卖出日，未完成交易按 2026-08-07 最新收盘价暂估。",
        "",
        "| 线 | 交易数 | 已完成 | 未完成暂估 | 胜率 | 平均单笔 | 收益率之和 | 每笔100元盈亏 | 每笔1000元盈亏 | 最好单笔 | 最差单笔 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"| {r.line} | {r.trades} | {r.completed} | {r.open_mark_to_market} | {r.win_rate:.2%} | {r.avg_ret:.2%} | {r.sum_ret:.2%} | {r.profit_if_100_each:.2f} | {r.profit_if_1000_each:.2f} | {r.best_trade:.2%} | {r.worst_trade:.2%} |"
        )
    lines.extend(["", "## 明细", ""])
    lines.append("| 线 | ETF | 信号日 | 买入日 | 结算日 | 状态 | 收益 | 每100元盈亏 |")
    lines.append("|---|---|---:|---:|---:|---|---:|---:|")
    for r in detail.sort_values(["line", "entry_date"]).itertuples(index=False):
        lines.append(
            f"| {r.line} | {r.symbol} {r.name} | {pd.Timestamp(r.signal_date).date()} | {pd.Timestamp(r.entry_date).date()} | {pd.Timestamp(r.settle_date).date()} | {r.status} | {r.ret:.2%} | {r.profit_if_100:.2f} |"
        )
    (OUT / "三线单独近一月复盘.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    print(OUT)


if __name__ == "__main__":
    main()
