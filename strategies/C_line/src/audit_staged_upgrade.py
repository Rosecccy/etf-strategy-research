from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import controlled_upgrade_search as search
import take_profit_10_test as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "staged_upgrade_refine"
SUMMARY_PATH = OUT / "summary.json"
EPS = 1e-9


def as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def next_trade_date(data: pd.DataFrame, date: pd.Timestamp) -> pd.Timestamp | None:
    later = data.loc[data["date"].gt(date), "date"]
    return pd.Timestamp(later.iloc[0]) if len(later) else None


def audit_line(spec: core.LineSpec) -> tuple[dict, pd.DataFrame]:
    path = OUT / f"{spec.key.lower()}_winner_trades.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"symbol": str})
    frame["symbol"] = frame["symbol"].str.zfill(6)
    date_columns = [
        "entry_date",
        "original_exit_date",
        "exit_date_test",
        "control_signal_date",
    ]
    for column in date_columns:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    prices = core.load_prices(spec.project, set(frame["symbol"]))
    rows: list[dict] = []

    for index, trade in frame.iterrows():
        symbol = str(trade["symbol"])
        data = prices[symbol]
        by_date = data.set_index("date")
        entry_date = pd.Timestamp(trade["entry_date"])
        exit_date = pd.Timestamp(trade["exit_date_test"])
        reason = str(trade.get("exit_reason_test", ""))
        signal_date = pd.to_datetime(trade.get("control_signal_date"), errors="coerce")
        errors: list[str] = []

        if entry_date not in by_date.index:
            errors.append("entry_date_missing")
        elif not np.isclose(
            float(trade["entry_close"]), float(by_date.loc[entry_date, "close"]), atol=EPS
        ):
            errors.append("entry_price_mismatch")

        if exit_date not in by_date.index:
            errors.append("exit_date_missing")
        elif reason.startswith("fixed_target"):
            target = float(trade["entry_close"]) * 1.20
            raw = by_date.loc[exit_date]
            expected = max(target, float(raw["open"]))
            if float(raw["high"]) + EPS < target:
                errors.append("target_not_touched")
            if not np.isclose(float(trade["exit_close_test"]), expected, atol=EPS):
                errors.append("target_fill_mismatch")
            if pd.isna(signal_date) or signal_date != exit_date:
                errors.append("target_signal_date_mismatch")
        else:
            raw_close = float(by_date.loc[exit_date, "close"])
            if not np.isclose(float(trade["exit_close_test"]), raw_close, atol=EPS):
                errors.append("exit_price_mismatch")
            if reason != "original_exit" and reason != "open_mark":
                expected = next_trade_date(data, pd.Timestamp(signal_date))
                if pd.isna(signal_date) or expected != exit_date:
                    errors.append("not_next_close_execution")

        original_exit = pd.to_datetime(trade.get("original_exit_date"), errors="coerce")
        if exit_date < entry_date:
            errors.append("exit_before_entry")
        if pd.notna(original_exit) and exit_date > original_exit:
            errors.append("exit_after_original")

        rows.append(
            {
                "line": spec.key,
                "row": int(index + 1),
                "symbol": symbol,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "reason": reason,
                "valid": not errors,
                "errors": "|".join(errors),
            }
        )

    detail = pd.DataFrame(rows)
    ordered = frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)
    overlaps = 0
    for index in range(1, len(ordered)):
        previous_exit = pd.Timestamp(ordered.loc[index - 1, "exit_date_test"])
        current_entry = pd.Timestamp(ordered.loc[index, "entry_date"])
        if current_entry < previous_exit:
            overlaps += 1

    changed = (
        pd.to_datetime(frame["exit_date_test"], errors="coerce")
        < pd.to_datetime(frame["original_exit_date"], errors="coerce")
    )
    stage1_flags = frame.get("stage1_triggered", pd.Series(False, index=frame.index)).map(as_bool)
    stage2_flags = frame.get("control_triggered", pd.Series(False, index=frame.index)).map(as_bool)
    effective_changed = changed | stage1_flags | stage2_flags
    payload = {
        "line": spec.key,
        "trades": int(len(frame)),
        "valid_trades": int(detail["valid"].sum()),
        "invalid_trades": int((~detail["valid"]).sum()),
        "overlaps": int(overlaps),
        "changed_exits": int(effective_changed.sum()),
        "stage1_changed": int(stage1_flags.sum()),
        "stage2_changed": int(stage2_flags.sum()),
    }
    return payload, detail


def pct_change(current: float, baseline: float) -> str:
    return f"{current / baseline - 1:+.2%}"


def make_report(summary: dict, audit: dict) -> str:
    lines = [
        "# C / S / D 控制变量升级审计",
        "",
        "## 口径",
        "",
        "- 原策略的买点、仓位与候选池保持不变，只改变卖出覆盖层。",
        "- 收盘条件在当日收盘确认，并在下一交易日收盘执行。",
        "- D 线固定止盈按可成交限价处理：当日最高价触及目标后，以目标价与开盘价较高者成交。",
        "- 交易费用沿用双边万三、单笔最低 5 元、100 股整数倍。",
        "- 开发段截至 2023 年；2024—2026 单独列为验证段。验证段已参与本轮候选门槛判断，因此当前仅为影子候选。",
        "",
        "## 结果",
        "",
        "| 线 | 升级规则 | 全期终值 | 全期提升 | 胜率 | 年均账户收益 | 最大回撤 | 2024—2026终值 | 验证段提升 | 邻域通过数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    audit_by_line = {item["line"]: item for item in audit["lines"]}
    for item in summary["results"]:
        base = item["baseline"]
        win = item["winner"]
        rule = item["winner_rule"]
        if item.get("stage1_rule"):
            rule = f"{item['stage1_rule']} + {rule}"
        lines.append(
            f"| {item['line']} | `{rule}` | {win['full']['final_value']:,.2f} | "
            f"{pct_change(win['full']['final_value'], base['full']['final_value'])} | "
            f"{win['full']['win_rate']:.2%} | {win['full']['avg_annual_return']:.2%} | "
            f"{win['full']['max_drawdown']:.2%} | {win['holdout']['final_value']:,.2f} | "
            f"{pct_change(win['holdout']['final_value'], base['holdout']['final_value'])} | "
            f"{item['strict_nearby_candidates']} |"
        )
    lines += [
        "",
        "## 成交审计",
        "",
        "| 线 | 逐笔有效 | 无效 | 提前退出 | 重叠交易 | 结论 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for line in ("C", "S", "D"):
        item = audit_by_line[line]
        conclusion = "通过" if item["invalid_trades"] == 0 and item["overlaps"] == 0 else "未通过"
        lines.append(
            f"| {line} | {item['valid_trades']}/{item['trades']} | {item['invalid_trades']} | "
            f"{item['changed_exits']} | {item['overlaps']} | {conclusion} |"
        )
    lines += [
        "",
        "## 当前判断",
        "",
        "三条线均找到了收益提高、胜率不下降且邻域中仍有效的规则。它们具备日线实盘执行口径，"
        "但由于 2024—2026 已被用于本轮研究筛选，暂时只能作为影子升级候选；正式替换前仍需按年度扩展窗口重选，或从下一交易日起冻结参数做前瞻观察。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    core.configure_stdout()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    payloads = []
    details = []
    for spec in core.SPECS:
        payload, detail = audit_line(spec)
        payloads.append(payload)
        details.append(detail)
    audit = {"lines": payloads}
    pd.concat(details, ignore_index=True).to_csv(
        OUT / "audit_detail.csv", index=False, encoding="utf-8-sig"
    )
    (OUT / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "REPORT.md").write_text(make_report(summary, audit), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
