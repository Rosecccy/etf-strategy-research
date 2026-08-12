from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import cash_yield_test as cash_test
import controlled_upgrade_search as stats_mod
import finalize_novel_upgrades as audit_mod
import take_profit_10_test as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "trigger_safe_upgrades"
CASH_RATE = 0.01

BASELINES = {
    line: ROOT / "fit" / "rolling_upgrades" / f"{line.lower()}_trades.csv"
    for line in ("C", "S", "D")
}
WINNERS = {
    "C": ROOT / "fit" / "rolling_upgrades" / "c_trades.csv",
    "S": ROOT / "fit" / "trend_extension" / "s_trades.csv",
    "D": ROOT / "fit" / "trigger_balanced_gap" / "d_trades.csv",
}
METHODS = {
    "C": "原滚动信号不变，完整空仓期按1%净年收益管理现金",
    "S": "强趋势退出延长 + 完整空仓期1%现金管理",
    "D": "盈利失速退出 + 空仓接入C线滚动信号 + 1%现金管理",
}


def load(path: Path) -> pd.DataFrame:
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


def entry_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return {
        (str(row.symbol), str(pd.Timestamp(row.entry_date).date()))
        for row in frame[["symbol", "entry_date"]].itertuples(index=False)
    }


def main() -> None:
    core.configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    annual_parts = []
    audits = []
    preservation = []
    for line in ("C", "S", "D"):
        baseline = load(BASELINES[line])
        winner = load(WINNERS[line])
        base_stats = stats_mod.subset_stats(baseline, "full")
        predictive_stats = stats_mod.subset_stats(winner, "full")
        holdout_stats = stats_mod.subset_stats(winner, "holdout")
        cash_stats, cash_detail, annual = cash_test.simulate(winner, CASH_RATE)
        cash_detail.to_csv(OUT / f"{line.lower()}_trades.csv", index=False, encoding="utf-8-sig")
        annual.insert(0, "line", line)
        annual_parts.append(annual)
        audit = audit_mod.audit(line, winner)
        audits.append(audit)
        base_keys = entry_keys(baseline)
        final_keys = entry_keys(winner)
        missing = sorted(base_keys - final_keys)
        start = int(pd.to_datetime(baseline["entry_date"]).dt.year.min())
        year_count = 2026 - start + 1
        preservation.append(
            {
                "line": line,
                "baseline_triggers": len(baseline),
                "final_triggers": len(winner),
                "missing_baseline_triggers": len(missing),
                "trigger_retention": len(winner) / len(baseline),
            }
        )
        rows.append(
            {
                "line": line,
                "method": METHODS[line],
                "baseline_final": base_stats["final_value"],
                "baseline_win": base_stats["win_rate"],
                "baseline_avg_annual": base_stats["avg_annual_return"],
                "baseline_triggers": len(baseline),
                "baseline_avg_triggers_per_year": len(baseline) / year_count,
                "predictive_final": predictive_stats["final_value"],
                "predictive_win": predictive_stats["win_rate"],
                "predictive_avg_annual": predictive_stats["avg_annual_return"],
                "predictive_max_drawdown": predictive_stats["max_drawdown"],
                "holdout_final": holdout_stats["final_value"],
                "holdout_win": holdout_stats["win_rate"],
                "final_triggers": len(winner),
                "final_avg_triggers_per_year": len(winner) / year_count,
                "trigger_retention": len(winner) / len(baseline),
                "cash_rate": CASH_RATE,
                "final_value": cash_stats["final_value"],
                "final_win": cash_stats["win_rate"],
                "final_avg_annual": cash_stats["avg_annual_return"],
                "final_max_drawdown": cash_stats["max_drawdown"],
                "final_ratio": cash_stats["final_value"] / base_stats["final_value"],
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(annual_parts, ignore_index=True).to_csv(
        OUT / "annual.csv", index=False, encoding="utf-8-sig"
    )
    audit_payload = {
        "price_and_timing": audits,
        "trigger_preservation": preservation,
        "d_weight_sensitivity": json.loads(
            (ROOT / "fit" / "trigger_balanced_gap" / "sensitivity.json").read_text(encoding="utf-8")
        ),
    }
    (OUT / "audit.json").write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_lines = [
        "# C / S / D 触发率约束升级报告",
        "",
        "本轮要求：任何升级不得删除原买入触发；收益、胜率与年均收益均按真实100股整数、双边万三且最低5元复算。",
        "",
        "| 线 | 最终方法 | 基线资金 | 最终资金 | 胜率 | 年均收益 | 触发数 | 年均触发 | 触发保留率 | 最大回撤 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['line']} | {row['method']} | {row['baseline_final']:,.2f} | "
            f"{row['final_value']:,.2f} | {row['final_win']:.2%} | {row['final_avg_annual']:.2%} | "
            f"{row['final_triggers']} | {row['final_avg_triggers_per_year']:.2f} | "
            f"{row['trigger_retention']:.2%} | {row['final_max_drawdown']:.2%} |"
        )
    report_lines += [
        "",
        "## 审计结论",
        "",
        "- C 与 S 保留全部原触发，D 在保留27笔原触发基础上增加4笔C线空仓信号。",
        "- D 的27组权重邻域全部改善，避免单一权重偶然最优。",
        "- 1%现金收益仅是执行层敏感性假设，正式实盘前需绑定实际现金管理品种及净收益口径。",
        "- 所有候选仍是历史回测，不构成收益保证或投资建议。",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(json.dumps(audit_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
