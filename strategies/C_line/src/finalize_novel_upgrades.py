from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import cash_yield_test as cash_test
import controlled_upgrade_search as search
import take_profit_10_test as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "novel_upgrades"
RATE = 0.01


SOURCES = {
    "C": ROOT / "fit" / "rolling_upgrades" / "c_trades.csv",
    "S": ROOT / "fit" / "trend_extension" / "s_trades.csv",
    "D": ROOT / "fit" / "novel_path_exits" / "d_trades.csv",
}


METHODS = {
    "C": "空仓现金管理：净年收益按1%计，原买卖信号完全不变",
    "S": "强趋势延长退出：趋势破坏后下一交易日收盘卖出",
    "D": "盈利后失速退出：达到3%浮盈且回落至盈亏平衡附近，下一交易日收盘卖出",
}


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"symbol": str})
    frame["symbol"] = frame["symbol"].str.zfill(6)
    for column in (
        "entry_date",
        "exit_date_test",
        "control_signal_date",
        "extend_signal_date",
        "path_signal_date",
    ):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def project(line: str) -> Path:
    return ROOT.parent / "S" if line == "S" else ROOT


def next_date(data: pd.DataFrame, date: pd.Timestamp) -> pd.Timestamp | None:
    later = data.loc[data["date"].gt(date), "date"]
    return pd.Timestamp(later.iloc[0]) if len(later) else None


def as_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def audit(line: str, frame: pd.DataFrame) -> dict:
    prices = core.load_prices(project(line), set(frame["symbol"]))
    errors = []
    for index, trade in frame.iterrows():
        data = prices[str(trade["symbol"])]
        entry_date = pd.Timestamp(trade["entry_date"])
        exit_date = pd.Timestamp(trade["exit_date_test"])
        entry_row = data[data["date"].eq(entry_date)]
        exit_row = data[data["date"].eq(exit_date)]
        local_errors = []
        if entry_row.empty:
            local_errors.append("entry_date_missing")
        elif not np.isclose(float(entry_row.iloc[0]["close"]), float(trade["entry_close"]), atol=1e-9):
            local_errors.append("entry_price")
        if exit_row.empty:
            local_errors.append("exit_date_missing")
        else:
            reason = str(trade.get("exit_reason_test", ""))
            target_match = re.search(r"fixed_target:a\+([0-9.]+)", reason)
            if target_match:
                target = float(trade["entry_close"]) * (1.0 + float(target_match.group(1)))
                expected = max(target, float(exit_row.iloc[0]["open"]))
                if float(exit_row.iloc[0]["high"]) + 1e-9 < target:
                    local_errors.append("target_not_touched")
            else:
                expected = float(exit_row.iloc[0]["close"])
            if not np.isclose(float(trade["exit_close_test"]), expected, atol=1e-9):
                local_errors.append("exit_price")
        signal = pd.NaT
        if as_bool(trade.get("extend_triggered", False)):
            signal = pd.to_datetime(trade.get("extend_signal_date"), errors="coerce")
        elif as_bool(trade.get("path_triggered", False)):
            signal = pd.to_datetime(trade.get("path_signal_date"), errors="coerce")
        elif as_bool(trade.get("control_triggered", False)) and "fixed_target" not in str(trade.get("exit_reason_test", "")):
            signal = pd.to_datetime(trade.get("control_signal_date"), errors="coerce")
        if not pd.isna(signal) and next_date(data, pd.Timestamp(signal)) != exit_date:
            local_errors.append("not_next_trade_close")
        if exit_date < entry_date:
            local_errors.append("chronology")
        if local_errors:
            errors.append(
                {
                    "line": line,
                    "row": index + 1,
                    "symbol": str(trade["symbol"]),
                    "entry_date": str(entry_date.date()),
                    "errors": "|".join(local_errors),
                }
            )
    overlaps = 0
    for index in range(1, len(frame)):
        if pd.Timestamp(frame.loc[index, "entry_date"]) < pd.Timestamp(frame.loc[index - 1, "exit_date_test"]):
            overlaps += 1
    return {
        "line": line,
        "trades": len(frame),
        "valid": len(frame) - len(errors),
        "invalid": len(errors),
        "overlaps": overlaps,
        "errors": errors,
    }


def report(metrics: pd.DataFrame, audits: list[dict]) -> str:
    lines = [
        "# C / S / D 新增升级研究",
        "",
        "所有预测规则均按年度扩展窗口，只使用测试年之前的数据。C线现金管理不改变预测信号，按净年收益1%进行保守情景核算。",
        "",
        "| 线 | 独立改进 | 原期末资金 | 改进后期末资金 | 再含1%现金管理 | 胜率 | 年度平均盈利率 | 最大回撤 | 预测规则2024-2026提升 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['line']} | {row['method']} | {row['baseline_final']:,.2f} | "
            f"{row['predictive_final']:,.2f} | {row['cash_final']:,.2f} | "
            f"{row['cash_win']:.2%} | {row['cash_avg_annual']:.2%} | {row['cash_max_drawdown']:.2%} | "
            f"{row['predictive_holdout_ratio'] - 1:+.2%} |"
        )
    lines += [
        "",
        "## 审计",
        "",
        "| 线 | 有效交易 | 无效交易 | 重叠持仓 |",
        "|---|---:|---:|---:|",
    ]
    for item in audits:
        lines.append(f"| {item['line']} | {item['valid']}/{item['trades']} | {item['invalid']} | {item['overlaps']} |")
    lines += [
        "",
        "## 边界",
        "",
        "- C线新增部分是现金利用率提升，不是择时能力提升；若实际净现金收益低于1%，应按敏感性表下调预期。",
        "- S线和D线的改进是预测/退出规则本身的滚动样本外提升，现金管理只是可选的共同执行层。",
        "- 现金管理只计算完整空仓间隔；持仓期间的零碎现金、首次交易前现金均不计息，属于偏保守口径。",
        "- 本报告不构成投资建议。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    core.configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    current_paths = {
        line: ROOT / "fit" / "rolling_upgrades" / f"{line.lower()}_trades.csv"
        for line in ("C", "S", "D")
    }
    metrics_rows = []
    annual_parts = []
    audits = []
    for line in ("C", "S", "D"):
        baseline_frame = load_frame(current_paths[line])
        candidate = load_frame(SOURCES[line])
        baseline = {period: search.subset_stats(baseline_frame, period) for period in ("full", "holdout")}
        predictive = {period: search.subset_stats(candidate, period) for period in ("full", "holdout")}
        cash_stats, cash_detail, annual = cash_test.simulate(candidate, RATE)
        cash_detail.to_csv(OUT / f"{line.lower()}_trades.csv", index=False, encoding="utf-8-sig")
        annual.insert(0, "line", line)
        annual_parts.append(annual)
        audits.append(audit(line, candidate))
        metrics_rows.append(
            {
                "line": line,
                "method": METHODS[line],
                "baseline_final": baseline["full"]["final_value"],
                "baseline_win": baseline["full"]["win_rate"],
                "baseline_avg_annual": baseline["full"]["avg_annual_return"],
                "predictive_final": predictive["full"]["final_value"],
                "predictive_win": predictive["full"]["win_rate"],
                "predictive_avg_annual": predictive["full"]["avg_annual_return"],
                "predictive_max_drawdown": predictive["full"]["max_drawdown"],
                "predictive_ratio": predictive["full"]["final_value"] / baseline["full"]["final_value"],
                "predictive_holdout_ratio": predictive["holdout"]["final_value"] / baseline["holdout"]["final_value"],
                "cash_net_yield": RATE,
                "cash_final": cash_stats["final_value"],
                "cash_win": cash_stats["win_rate"],
                "cash_avg_annual": cash_stats["avg_annual_return"],
                "cash_max_drawdown": cash_stats["max_drawdown"],
                "idle_days": cash_stats["idle_days"],
            }
        )
    metrics = pd.DataFrame(metrics_rows)
    annual_table = pd.concat(annual_parts, ignore_index=True)
    metrics.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    annual_table.to_csv(OUT / "annual.csv", index=False, encoding="utf-8-sig")
    (OUT / "audit.json").write_text(json.dumps({"lines": audits}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "REPORT.md").write_text(report(metrics, audits), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(json.dumps({"lines": audits}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
