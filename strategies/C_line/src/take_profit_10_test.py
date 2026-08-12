from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OUT = ROOT / "fit" / "take_profit_10"
INITIAL_CAPITAL = 10_000.0
COMMISSION_RATE = 0.0003
MIN_COMMISSION = 5.0
TARGET_RETURN = 0.10
AS_OF = pd.Timestamp("2026-07-31")


@dataclass(frozen=True)
class LineSpec:
    key: str
    trade_path: Path
    project: Path
    exit_date_col: str
    exit_price_col: str
    status_col: str | None = None


SPECS = (
    LineSpec(
        "C",
        ROOT / "fit" / "clean_upgrades" / "c_trades.csv",
        ROOT,
        "exit_date",
        "exit_close",
    ),
    LineSpec(
        "S",
        ROOT / "fit" / "clean_upgrades" / "s_trades.csv",
        WORKSPACE / "S",
        "exit_date",
        "exit_close",
    ),
    LineSpec(
        "D",
        WORKSPACE / "D" / "out" / "clean_control_sweep" / "strong_exit_d3_pos_trades.csv",
        ROOT,
        "actual_exit_date",
        "actual_exit_close",
        "status_portfolio",
    ),
)


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def commission(notional: float) -> float:
    return max(MIN_COMMISSION, abs(notional) * COMMISSION_RATE)


def raw_path(project: Path, symbol: str) -> Path:
    path = project / "raw" / "etf" / f"{symbol}.csv"
    if not path.exists() and project.name == "S":
        path = ROOT / "raw" / "etf" / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw data for {symbol}: {path}")
    return path


def load_prices(project: Path, symbols: set[str]) -> dict[str, pd.DataFrame]:
    result = {}
    for symbol in sorted(symbols):
        frame = pd.read_csv(raw_path(project, symbol), encoding="utf-8-sig")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in ("open", "high", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        result[symbol] = (
            frame.dropna(subset=["date", "open", "high", "close"])
            .sort_values("date")
            .drop_duplicates("date", keep="last")
            .reset_index(drop=True)
        )
    return result


def load_trades(spec: LineSpec) -> pd.DataFrame:
    frame = pd.read_csv(spec.trade_path, encoding="utf-8-sig", dtype={"symbol": str})
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    frame[spec.exit_date_col] = pd.to_datetime(frame[spec.exit_date_col], errors="coerce")
    frame["entry_close"] = pd.to_numeric(frame["entry_close"], errors="coerce")
    frame[spec.exit_price_col] = pd.to_numeric(frame[spec.exit_price_col], errors="coerce")
    frame = frame.dropna(subset=["entry_date", "entry_close"]).copy()
    frame["original_exit_date"] = frame[spec.exit_date_col]
    frame["original_exit_close"] = frame[spec.exit_price_col]
    if spec.status_col:
        frame["original_open"] = frame[spec.status_col].ne("closed") | frame[spec.exit_date_col].isna()
    else:
        frame["original_open"] = False
    return frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def apply_take_profit(
    trades: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    mode: str,
) -> pd.DataFrame:
    rows = []
    for trade_number, trade in trades.iterrows():
        item = trade.to_dict()
        data = prices[str(trade["symbol"])]
        entry_date = pd.Timestamp(trade["entry_date"])
        end_date = (
            AS_OF
            if bool(trade["original_open"])
            else pd.Timestamp(trade["original_exit_date"])
        )
        path = data[data["date"].gt(entry_date) & data["date"].le(end_date)]
        target = float(trade["entry_close"]) * (1.0 + TARGET_RETURN)
        if mode == "intraday_limit":
            hits = path[path["high"].ge(target)]
        elif mode == "close_confirmed":
            hits = path[path["close"].ge(target)]
        elif mode == "baseline":
            hits = path.iloc[0:0]
        else:
            raise ValueError(mode)

        item["trade_number"] = int(trade_number + 1)
        item["take_profit_mode"] = mode
        item["take_profit_triggered"] = False
        item["take_profit_price"] = target
        if not hits.empty:
            hit = hits.iloc[0]
            item["take_profit_triggered"] = True
            item["exit_date_test"] = pd.Timestamp(hit["date"])
            if mode == "intraday_limit":
                item["exit_close_test"] = max(target, float(hit["open"]))
                item["exit_reason_test"] = "intraday_limit_10pct"
            else:
                item["exit_close_test"] = float(hit["close"])
                item["exit_reason_test"] = "close_confirmed_10pct"
            item["open_mark_test"] = False
        elif bool(trade["original_open"]):
            mark = data[data["date"].le(AS_OF)].iloc[-1]
            item["exit_date_test"] = pd.Timestamp(mark["date"])
            item["exit_close_test"] = float(mark["close"])
            item["exit_reason_test"] = "open_mark"
            item["open_mark_test"] = True
        else:
            item["exit_date_test"] = pd.Timestamp(trade["original_exit_date"])
            item["exit_close_test"] = float(trade["original_exit_close"])
            item["exit_reason_test"] = "original_exit"
            item["open_mark_test"] = False
        item["gross_return_test"] = (
            float(item["exit_close_test"]) / float(trade["entry_close"]) - 1.0
        )
        rows.append(item)
    return pd.DataFrame(rows)


def simulate_account(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cash = INITIAL_CAPITAL
    trade_rows = []
    for _, trade in frame.sort_values(["entry_date", "symbol"]).iterrows():
        before = cash
        entry = float(trade["entry_close"])
        exit_price = float(trade["exit_close_test"])
        quantity = math.floor(cash / (entry * 100)) * 100
        while quantity >= 100 and quantity * entry + commission(quantity * entry) > cash:
            quantity -= 100
        if quantity < 100:
            continue
        buy_value = quantity * entry
        buy_fee = commission(buy_value)
        cash -= buy_value + buy_fee
        if bool(trade["open_mark_test"]):
            mark_value = quantity * exit_price
            final_value = cash + mark_value - commission(mark_value)
            account_return = final_value / before - 1.0
            after = final_value
        else:
            sell_value = quantity * exit_price
            sell_fee = commission(sell_value)
            cash += sell_value - sell_fee
            after = cash
            account_return = after / before - 1.0
        item = trade.to_dict()
        item.update(
            {
                "quantity_test": int(quantity),
                "buy_fee_test": float(buy_fee),
                "cash_before_test": float(before),
                "cash_after_test": float(after),
                "account_return_test": float(account_return),
            }
        )
        trade_rows.append(item)

    detail = pd.DataFrame(trade_rows)
    final_value = float(detail.iloc[-1]["cash_after_test"]) if len(detail) else INITIAL_CAPITAL
    returns = detail["account_return_test"].astype(float)
    equity = INITIAL_CAPITAL * (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0 if len(equity) else pd.Series(dtype=float)
    detail["year"] = pd.to_datetime(detail["entry_date"]).dt.year
    annual = (
        detail.groupby("year")["account_return_test"]
        .apply(lambda values: float(np.prod(1.0 + values.astype(float)) - 1.0))
        .reindex(range(int(detail["year"].min()), 2027), fill_value=0.0)
        .rename("account_return")
        .reset_index()
    )
    completed = detail[~detail["open_mark_test"]]
    stats = {
        "trades": int(len(detail)),
        "closed_trades": int(len(completed)),
        "final_value": final_value,
        "total_return": final_value / INITIAL_CAPITAL - 1.0,
        "win_rate": float((completed["account_return_test"] > 0).mean()) if len(completed) else 0.0,
        "avg_trade_return": float(completed["account_return_test"].mean()) if len(completed) else 0.0,
        "avg_annual_return": float(annual["account_return"].mean()),
        "positive_year_rate": float((annual["account_return"] > 0).mean()),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "avg_holding_days": float(
            (pd.to_datetime(detail["exit_date_test"]) - pd.to_datetime(detail["entry_date"])).dt.days.mean()
        ),
        "take_profit_trades": int(detail["take_profit_triggered"].sum()),
    }
    return stats, detail, annual


def evaluate_line(spec: LineSpec) -> dict:
    trades = load_trades(spec)
    prices = load_prices(spec.project, set(trades["symbol"]))
    results = {}
    for mode in ("baseline", "intraday_limit", "close_confirmed"):
        adjusted = apply_take_profit(trades, prices, mode)
        stats, detail, annual = simulate_account(adjusted)
        holdout = adjusted[pd.to_datetime(adjusted["entry_date"]).dt.year.ge(2024)].copy()
        holdout_stats, _, _ = simulate_account(holdout)
        stats["holdout_2024_2026"] = holdout_stats
        detail.to_csv(OUT / f"{spec.key.lower()}_{mode}_trades.csv", index=False, encoding="utf-8-sig")
        annual.to_csv(OUT / f"{spec.key.lower()}_{mode}_annual.csv", index=False, encoding="utf-8-sig")
        results[mode] = stats
    return {"line": spec.key, "results": results}


def make_report(summary: dict) -> str:
    lines = [
        "# C / S / D 固定 +10% 止盈测试",
        "",
        "- 控制变量：买点、候选排序和原卖出规则全部不变，仅加入固定止盈。",
        "- 盘中限价：买入后的交易日最高价首次达到买入价 × 1.10，即按 10% 限价成交；若跳空高开超过目标，则按开盘价成交。",
        "- 收盘确认：收盘价首次达到买入价 × 1.10，当日收盘卖出。",
        "- 成本：万三、单边最低 5 元、100 股整数手。",
        "",
        "| 线 | 口径 | 触发笔数 | 全期终值 | 全期胜率 | 2024-2026终值 | 2024-2026胜率 | 最大回撤 | 平均持有天数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"baseline": "原策略", "intraday_limit": "盘中触及10%", "close_confirmed": "收盘达到10%"}
    for line in summary["lines"]:
        for mode, stats in line["results"].items():
            lines.append(
                f"| {line['line']} | {labels[mode]} | {stats['take_profit_trades']} | "
                f"{stats['final_value']:.2f} | {stats['win_rate']:.2%} | "
                f"{stats['holdout_2024_2026']['final_value']:.2f} | "
                f"{stats['holdout_2024_2026']['win_rate']:.2%} | {stats['max_drawdown']:.2%} | "
                f"{stats['avg_holding_days']:.1f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [evaluate_line(spec) for spec in SPECS]
    summary = {
        "target_return": TARGET_RETURN,
        "execution": {
            "intraday_limit": "first post-entry day high >= entry * 1.10; fill at max(target, open)",
            "close_confirmed": "first post-entry day close >= entry * 1.10; fill at close",
            "fees": "0.03% each side, minimum CNY 5, 100-share lots",
        },
        "lines": lines,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "REPORT.md").write_text(make_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
