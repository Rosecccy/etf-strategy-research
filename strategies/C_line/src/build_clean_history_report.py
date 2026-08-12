from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OUT = ROOT / "fit" / "formal_clean"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def pct(value: float | None) -> str:
    return "-" if value is None else f"{float(value):.2%}"


def money(value: float | None) -> str:
    return "-" if value is None else f"{float(value):,.2f}"


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    c = read_json(OUT / "summary.json")
    s = read_json(WORKSPACE / "S" / "fit" / "selector" / "summary.json")
    d = read_json(WORKSPACE / "D" / "out" / "formal_clean" / "summary.json")
    s_official = s["execution_corrected"]

    rows = [
        {
            "line": "C",
            "version": c["version"],
            "universe": c["approved_universe"],
            "period": "2014-2026",
            "initial_capital": 10000.0,
            "final_capital": c["net_10000"]["final_cash"],
            "total_return": c["net_10000"]["final_cash"] / 10000.0 - 1.0,
            "trades": c["all"]["trades"],
            "win_rate": c["all"]["win_rate"],
            "max_drawdown": c["all"]["max_drawdown"],
            "holdout_return": c["holdout"]["final_1000"] / 1000.0 - 1.0,
            "open_positions": 0,
            "execution": "T+1收盘；万三、单边最低5元；100股整数手",
        },
        {
            "line": "S",
            "version": read_json(WORKSPACE / "S" / "cfg" / "s1_model.json").get("version", "S1-clean"),
            "universe": s["approved_universe"],
            "period": "2014-2026",
            "initial_capital": 10000.0,
            "final_capital": s_official["net_10000"]["final_cash"],
            "total_return": s_official["net_10000"]["final_cash"] / 10000.0 - 1.0,
            "trades": s_official["all"]["trades"],
            "win_rate": s_official["all"]["win_rate"],
            "max_drawdown": s_official["all"]["max_drawdown"],
            "holdout_return": s_official["holdout"]["final_1000"] / 1000.0 - 1.0,
            "open_positions": 0,
            "execution": "T+1收盘；万三、单边最低5元；100股整数手",
        },
        {
            "line": "D",
            "version": read_json(WORKSPACE / "D" / "cfg" / "daily_panic.json").get("version", "D3-clean"),
            "universe": d["approved_universe"],
            "period": "2019-2026",
            "initial_capital": d["all"]["initial_capital"],
            "final_capital": d["all"]["final_value"],
            "total_return": d["all"]["total_return"],
            "trades": d["all"]["closed_trades"],
            "win_rate": d["all"]["win_rate"],
            "max_drawdown": d["all"]["max_drawdown"],
            "holdout_return": d["holdout"]["total_return"],
            "open_positions": d["all"]["open_positions"],
            "execution": "T+1收盘；万三、单边最低5元；100股整数手",
        },
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "csd_summary.csv", index=False, encoding="utf-8-sig")

    annual_parts = []
    for line, path, return_col, return_type in (
        ("C", OUT / "annual.csv", "gross_return", "逐笔复合收益（未扣费）"),
        ("S", WORKSPACE / "S" / "fit" / "selector" / "annual.csv", "gross_return", "逐笔复合收益（未扣费）"),
        ("D", WORKSPACE / "D" / "out" / "formal_clean" / "annual.csv", "compounded_net_return", "账户净收益（已扣费）"),
    ):
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame.insert(0, "line", line)
        frame["return_type"] = return_type
        if return_col != "strategy_return":
            frame = frame.rename(columns={return_col: "strategy_return"})
        annual_parts.append(frame)
    pd.concat(annual_parts, ignore_index=True, sort=False).to_csv(
        OUT / "csd_annual.csv", index=False, encoding="utf-8-sig"
    )

    lines = [
        "# C/S/D 干净数据滚动重训正式结论",
        "",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "口径：只使用质量闸门批准的 ETF；每个测试年只使用此前已完成的数据选参数；信号收盘确认后按下一交易日收盘执行；正式账户按100股整数手、佣金万三且单边最低5元计算。",
        "",
        "| 策略线 | 正式期 | ETF数 | 初始资金 | 期末资金 | 总收益 | 已平仓 | 胜率 | 最大回撤 | 2024-2026样本外收益 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['line']} | {row['period']} | {row['universe']} | {money(row['initial_capital'])} | "
            f"{money(row['final_capital'])} | {pct(row['total_return'])} | {row['trades']} | "
            f"{pct(row['win_rate'])} | {pct(row['max_drawdown'])} | {pct(row['holdout_return'])} |"
        )
    lines.extend(
        [
            "",
            "## 治理结论",
            "",
            "- C/S 混合旧方案在干净数据重训后未通过样本外晋级门槛，正式 C 线回退为纯 C 主策略。",
            "- S 线已排除异常 ETF 512690，并对主策略抢占空仓补偿仓位的卖出时点做了实盘校正。",
            "- D 线使用逐年过去数据选择恐慌阈值与滞涨退出规则；当前未平仓按最新收盘价计入期末权益。",
            "- `csd_annual.csv` 中 C/S 年度收益为未扣费的逐笔复合诊断值；最终资金才是正式扣费账户结论。D 年度表为扣费账户收益。",
            "",
            "## 可复核文件",
            "",
            "- `C/raw/quality.json` 与 `S/raw/quality.json`：数据质量闸门。",
            "- `C/fit/formal_clean/`：C 正式交易、年度结果及总表。",
            "- `S/fit/selector/formal_execution_trades.csv`：S 正式成交时序校正交易。",
            "- `D/out/formal_clean/`：D 正式滚动选择、交易和年度结果。",
        ]
    )
    (OUT / "FORMAL_HISTORY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
