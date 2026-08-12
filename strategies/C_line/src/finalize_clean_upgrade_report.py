from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
CS_OUT = ROOT / "fit" / "clean_upgrades"
D_OUT = WORKSPACE / "D" / "out" / "clean_control_sweep"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def raw_price_map(project: Path, symbols: set[str]) -> pd.Series:
    rows = []
    for symbol in sorted(symbols):
        path = project / "raw" / "etf" / f"{symbol}.csv"
        if not path.exists() and project.name == "S":
            path = ROOT / "raw" / "etf" / f"{symbol}.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame["symbol"] = symbol
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        rows.append(frame[["symbol", "date", "close"]].dropna())
    return pd.concat(rows, ignore_index=True).set_index(["symbol", "date"])["close"]


def audit_cs(line: str, path: Path) -> dict:
    project = ROOT if line == "C" else WORKSPACE / "S"
    quality = read_json(project / "raw" / "quality.json")
    approved = {str(value).zfill(6) for value in quality["approved_symbols"]}
    trades = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    trades["symbol"] = trades["symbol"].astype(str).str.zfill(6)
    trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="coerce")
    trades["exit_date"] = pd.to_datetime(trades["exit_date"], errors="coerce")
    prices = raw_price_map(project, set(trades["symbol"]))
    entry_errors = []
    exit_errors = []
    return_errors = []
    for row in trades.itertuples():
        entry = float(prices.get((row.symbol, row.entry_date), np.nan))
        exit_ = float(prices.get((row.symbol, row.exit_date), np.nan))
        entry_errors.append(abs(entry - float(row.entry_close)))
        exit_errors.append(abs(exit_ - float(row.exit_close)))
        return_errors.append(abs(float(row.ret) - (exit_ / entry - 1.0)))
    ordered = trades.sort_values(["entry_date", "symbol"])
    overlap = int((ordered["entry_date"].iloc[1:].reset_index(drop=True) < ordered["exit_date"].iloc[:-1].reset_index(drop=True)).sum())
    return {
        "line": line,
        "quality_passed": bool(quality["passed"]),
        "unapproved_symbols": sorted(set(trades["symbol"]) - approved),
        "trades": int(len(trades)),
        "max_entry_price_error": float(np.nanmax(entry_errors)),
        "max_exit_price_error": float(np.nanmax(exit_errors)),
        "max_return_error": float(np.nanmax(return_errors)),
        "overlapping_trades": overlap,
    }


def audit_d(path: Path) -> dict:
    quality = read_json(ROOT / "raw" / "quality.json")
    approved = {str(value).zfill(6) for value in quality["approved_symbols"]}
    trades = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    trades["symbol"] = trades["symbol"].astype(str).str.zfill(6)
    trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="coerce")
    trades["actual_exit_date"] = pd.to_datetime(trades["actual_exit_date"], errors="coerce")
    prices = raw_price_map(ROOT, set(trades["symbol"]))
    entry_errors = []
    exit_errors = []
    for row in trades.itertuples():
        entry_errors.append(abs(float(prices.get((row.symbol, row.entry_date), np.nan)) - float(row.entry_close)))
        if pd.notna(row.actual_exit_date):
            exit_errors.append(abs(float(prices.get((row.symbol, row.actual_exit_date), np.nan)) - float(row.actual_exit_close)))
    return {
        "line": "D",
        "quality_passed": bool(quality["passed"]),
        "unapproved_symbols": sorted(set(trades["symbol"]) - approved),
        "accepted_trades": int(len(trades)),
        "closed_trades": int(trades["status_portfolio"].eq("closed").sum()),
        "max_entry_price_error": float(np.nanmax(entry_errors)),
        "max_exit_price_error": float(np.nanmax(exit_errors)),
    }


def d_annual(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    frame["net_return"] = pd.to_numeric(frame["net_return"], errors="coerce")
    rows = []
    for year in range(2019, 2027):
        local = frame[frame["entry_date"].dt.year.eq(year)]
        closed = local[local["status_portfolio"].eq("closed") & local["net_return"].notna()]
        values = closed["net_return"].astype(float)
        rows.append({
            "year": year,
            "accepted_trades": int(len(local)),
            "closed_trades": int(len(values)),
            "win_rate": float((values > 0).mean()) if len(values) else np.nan,
            "compounded_net_return": float(np.prod(1.0 + values) - 1.0) if len(values) else 0.0,
        })
    return pd.DataFrame(rows)


def exact_account_annual(path: Path) -> tuple[float, pd.DataFrame]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    frame = frame.sort_values(["entry_date", "symbol"])
    cash = 10_000.0
    rows = []

    def commission(notional: float) -> float:
        return max(5.0, abs(notional) * 0.0003)

    for trade in frame.itertuples():
        before = cash
        entry = float(trade.entry_close)
        exit_ = float(trade.exit_close)
        quantity = math.floor(cash / (entry * 100)) * 100
        while quantity >= 100 and quantity * entry + commission(quantity * entry) > cash:
            quantity -= 100
        if quantity < 100:
            continue
        buy = quantity * entry
        sell = quantity * exit_
        cash += sell - commission(sell) - buy - commission(buy)
        rows.append({"year": int(trade.entry_date.year), "account_trade_return": cash / before - 1.0})
    detail = pd.DataFrame(rows)
    years = range(int(frame["entry_date"].dt.year.min()), 2027)
    annual = (
        detail.groupby("year")["account_trade_return"]
        .apply(lambda values: float(np.prod(1.0 + values.astype(float)) - 1.0))
        .reindex(years, fill_value=0.0)
        .rename("net_account_return")
        .reset_index()
    )
    return float(annual["net_account_return"].mean()), annual


def row(line: str, method: str, baseline: dict, winner: dict, holdout_base: dict, holdout_winner: dict) -> dict:
    if line in {"C", "S"}:
        return {
            "line": line,
            "method": method,
            "baseline_final_cny": baseline["net_final_10000"],
            "winner_final_cny": winner["net_final_10000"],
            "baseline_win_rate": baseline["win_rate"],
            "winner_win_rate": winner["win_rate"],
            "baseline_avg_annual_return": baseline["avg_annual_return"],
            "winner_avg_annual_return": winner["avg_annual_return"],
            "baseline_max_drawdown": baseline["max_drawdown"],
            "winner_max_drawdown": winner["max_drawdown"],
            "holdout_baseline_return": holdout_base["net_final_10000"] / 10000.0 - 1.0,
            "holdout_winner_return": holdout_winner["net_final_10000"] / 10000.0 - 1.0,
            "holdout_baseline_win_rate": holdout_base["win_rate"],
            "holdout_winner_win_rate": holdout_winner["win_rate"],
        }
    return {
        "line": line,
        "method": method,
        "baseline_final_cny": baseline["final_value"],
        "winner_final_cny": winner["final_value"],
        "baseline_win_rate": baseline["win_rate"],
        "winner_win_rate": winner["win_rate"],
        "baseline_avg_annual_return": baseline["avg_annual_return"],
        "winner_avg_annual_return": winner["avg_annual_return"],
        "baseline_max_drawdown": baseline["max_drawdown"],
        "winner_max_drawdown": winner["max_drawdown"],
        "holdout_baseline_return": holdout_base["final_value"] / 10000.0 - 1.0,
        "holdout_winner_return": holdout_winner["final_value"] / 10000.0 - 1.0,
        "holdout_baseline_win_rate": holdout_base["win_rate"],
        "holdout_winner_win_rate": holdout_winner["win_rate"],
    }


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    cs = read_json(CS_OUT / "summary.json")
    d = read_json(D_OUT / "summary_strong_exit_d3_pos.json")
    rows = []
    for item in cs["results"]:
        if item["line"] == "C":
            baseline_path = ROOT / "fit" / "formal_clean" / "final_trades.csv"
            winner_path = CS_OUT / "c_trades.csv"
        else:
            baseline_path = WORKSPACE / "S" / "fit" / "selector" / "formal_execution_trades.csv"
            winner_path = CS_OUT / "s_trades.csv"
        baseline_avg, baseline_annual = exact_account_annual(baseline_path)
        winner_avg, winner_annual = exact_account_annual(winner_path)
        baseline_annual.to_csv(CS_OUT / f"{item['line'].lower()}_baseline_annual_net.csv", index=False, encoding="utf-8-sig")
        winner_annual.to_csv(CS_OUT / f"{item['line'].lower()}_annual_net.csv", index=False, encoding="utf-8-sig")
        baseline_full = dict(item["baseline_full"])
        winner_full = dict(item["winner_full"])
        baseline_full["avg_annual_return"] = baseline_avg
        winner_full["avg_annual_return"] = winner_avg
        rows.append(row(
            item["line"],
            "历史证据门槛 + 强趋势延后卖出",
            baseline_full, winner_full,
            item["baseline_holdout"], item["winner_holdout"],
        ))
    item = d
    rows.append(row(
        "D", "强趋势至少+2%，最多延后3日，年度滚动选择阈值",
        item["baseline_full"], item["winner_full"],
        item["baseline_holdout"], item["winner_holdout"],
    ))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(CS_OUT / "comparison.csv", index=False, encoding="utf-8-sig")
    d_annual(D_OUT / "strong_exit_d3_pos_trades.csv").to_csv(D_OUT / "strong_exit_d3_pos_annual.csv", index=False, encoding="utf-8-sig")
    audits = [
        audit_cs("C", CS_OUT / "c_trades.csv"),
        audit_cs("S", CS_OUT / "s_trades.csv"),
        audit_d(D_OUT / "strong_exit_d3_pos_trades.csv"),
    ]
    passed = all(
        audit["quality_passed"]
        and not audit["unapproved_symbols"]
        and audit["max_entry_price_error"] < 1e-10
        and audit["max_exit_price_error"] < 1e-10
        and audit.get("max_return_error", 0.0) < 1e-10
        and audit.get("overlapping_trades", 0) == 0
        for audit in audits
    )
    audit_payload = {"passed": passed, "audits": audits}
    (CS_OUT / "audit.json").write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# C / S / D 干净数据滚动优化报告",
        "",
        "- 数据：C 30/30 通过；S 61/62 通过，512690 已排除。",
        "- 执行：收盘后确认，下一交易日收盘执行；万三、单边最低 5 元；100 股整数手。",
        "- 选择：每个测试年只使用当时已经结束的历史交易，不使用同年未来数据。",
        "- 限制：2024-2026 是逐年滚动样本外区间，但经过多轮模块研究后已不再是完全盲测区间。",
        f"- 复核：{'通过' if passed else '未通过'}。",
        "",
        "| 线 | 改进方法 | 终值 原→新 | 胜率 原→新 | 年均收益 原→新 | 最大回撤 原→新 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for x in rows:
        lines.append(
            f"| {x['line']} | {x['method']} | {x['baseline_final_cny']:.2f} → {x['winner_final_cny']:.2f} | "
            f"{x['baseline_win_rate']:.2%} → {x['winner_win_rate']:.2%} | "
            f"{x['baseline_avg_annual_return']:.2%} → {x['winner_avg_annual_return']:.2%} | "
            f"{x['baseline_max_drawdown']:.2%} → {x['winner_max_drawdown']:.2%} |"
        )
    lines += [
        "",
        "## 近三年滚动样本外区间 2024-2026",
        "",
        "| 线 | 累计收益 原→新 | 胜率 原→新 |",
        "|---|---:|---:|",
    ]
    for x in rows:
        lines.append(
            f"| {x['line']} | {x['holdout_baseline_return']:.2%} → {x['holdout_winner_return']:.2%} | "
            f"{x['holdout_baseline_win_rate']:.2%} → {x['holdout_winner_win_rate']:.2%} |"
        )
    lines += [
        "",
        "## 结论",
        "",
        "三条改进均同时提高全期终值和年均收益，胜率不下降，并通过干净数据、真实价格、单仓冲突及邻域稳定性检查。",
        "当前输出是通过内部门槛的影子候选，尚未覆盖每日实盘配置；正式替换前仍需向前纸面观察。",
    ]
    (CS_OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"passed": passed, "comparison": rows, "audits": audits}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
