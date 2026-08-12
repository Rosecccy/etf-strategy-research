from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import take_profit_10_test as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "rolling_upgrades"
EPS = 1e-9


SOURCES = {
    "C": ROOT / "fit" / "walkforward_overlay" / "c_trades.csv",
    "S": ROOT / "fit" / "s_frozen_2024" / "winner_trades.csv",
    "D": ROOT / "fit" / "d_robust_plateau" / "winner_trades.csv",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def result_payloads() -> dict[str, dict]:
    c = load_json(ROOT / "fit" / "walkforward_overlay" / "summary.json")
    c_result = next(item for item in c["results"] if item["line"] == "C")
    s = load_json(ROOT / "fit" / "s_frozen_2024" / "summary.json")
    d = load_json(ROOT / "fit" / "d_robust_plateau" / "summary.json")
    return {
        "C": {
            "method": "年度扩展窗口退出增强选择",
            "baseline": c_result["baseline"],
            "winner": c_result["walkforward"],
            "stability_count": 18,
            "rule": "每年仅用此前交易选择弱势退出/趋势回撤退出",
        },
        "S": {
            "method": "截至2023训练，2024起冻结",
            "baseline": s["baseline"],
            "winner": {
                "full": {
                    **s["baseline"]["full"],
                    "final_value": s["winner"]["full_final_value"],
                    "win_rate": s["winner"]["full_win_rate"],
                    "max_drawdown": s["winner"]["full_max_drawdown"],
                    "avg_annual_return": s["winner"]["full_avg_annual_return"],
                },
                "holdout": {
                    **s["baseline"]["holdout"],
                    "final_value": s["winner"]["holdout_final_value"],
                    "win_rate": s["winner"]["holdout_win_rate"],
                    "max_drawdown": s["winner"]["holdout_max_drawdown"],
                    "avg_annual_return": s["winner"]["holdout_avg_annual_return"],
                },
            },
            "stability_count": s["passing_neighbor_count"],
            "rule": s["winner"]["rule"],
        },
        "D": {
            "method": "年度扩展窗口近优止盈平台中位数",
            "baseline": d["baseline"],
            "winner": {
                "full": {
                    **d["baseline"]["full"],
                    "final_value": d["winner"]["full_final_value"],
                    "win_rate": d["winner"]["full_win_rate"],
                    "max_drawdown": d["winner"]["full_max_drawdown"],
                    "avg_annual_return": d["winner"]["full_avg_annual_return"],
                },
                "holdout": {
                    **d["baseline"]["holdout"],
                    "final_value": d["winner"]["holdout_final_value"],
                    "win_rate": d["winner"]["holdout_win_rate"],
                    "max_drawdown": d["winner"]["holdout_max_drawdown"],
                    "avg_annual_return": d["winner"]["holdout_avg_annual_return"],
                },
            },
            "stability_count": d["passing_neighbor_count"],
            "rule": f"近优平台宽度 {d['winner']['plateau_fraction']:.0%}，取目标中位数",
        },
    }


def raw_close(data: pd.DataFrame, date: pd.Timestamp) -> float:
    row = data[data["date"].eq(date)]
    if row.empty:
        raise KeyError(date)
    return float(row.iloc[0]["close"])


def next_date(data: pd.DataFrame, date: pd.Timestamp) -> pd.Timestamp | None:
    later = data.loc[data["date"].gt(date), "date"]
    return pd.Timestamp(later.iloc[0]) if len(later) else None


def target_from_reason(reason: str) -> float | None:
    match = re.search(r"fixed_target:a\+([0-9.]+)", reason)
    return float(match.group(1)) if match else None


def audit_line(spec: core.LineSpec, frame: pd.DataFrame) -> dict:
    prices = core.load_prices(spec.project, set(frame["symbol"]))
    invalid = []
    for index, trade in frame.iterrows():
        data = prices[str(trade["symbol"])]
        entry_date = pd.Timestamp(trade["entry_date"])
        exit_date = pd.Timestamp(trade["exit_date_test"])
        reason = str(trade.get("exit_reason_test", ""))
        errors = []
        try:
            if not np.isclose(raw_close(data, entry_date), float(trade["entry_close"]), atol=EPS):
                errors.append("entry_price")
        except KeyError:
            errors.append("entry_date")
        exit_row = data[data["date"].eq(exit_date)]
        if exit_row.empty:
            errors.append("exit_date")
        else:
            target = target_from_reason(reason)
            if target is not None:
                target_price = float(trade["entry_close"]) * (1.0 + target)
                row = exit_row.iloc[0]
                expected = max(target_price, float(row["open"]))
                if float(row["high"]) + EPS < target_price:
                    errors.append("target_not_touched")
                if not np.isclose(float(trade["exit_close_test"]), expected, atol=EPS):
                    errors.append("target_fill")
            else:
                expected = float(exit_row.iloc[0]["close"])
                if not np.isclose(float(trade["exit_close_test"]), expected, atol=EPS):
                    errors.append("exit_price")
                if reason not in {"original_exit", "open_mark"}:
                    signal = pd.to_datetime(trade.get("control_signal_date"), errors="coerce")
                    if pd.isna(signal) or next_date(data, pd.Timestamp(signal)) != exit_date:
                        errors.append("not_t1_close")
        if exit_date < entry_date:
            errors.append("chronology")
        if errors:
            invalid.append(
                {
                    "line": spec.key,
                    "row": int(index + 1),
                    "symbol": str(trade["symbol"]),
                    "errors": "|".join(errors),
                }
            )
    ordered = frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)
    overlaps = sum(
        pd.Timestamp(ordered.loc[index, "entry_date"])
        < pd.Timestamp(ordered.loc[index - 1, "exit_date_test"])
        for index in range(1, len(ordered))
    )
    return {
        "line": spec.key,
        "trades": int(len(frame)),
        "valid": int(len(frame) - len(invalid)),
        "invalid": int(len(invalid)),
        "overlaps": int(overlaps),
        "errors": invalid,
    }


def make_metrics(payloads: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for line, payload in payloads.items():
        base = payload["baseline"]
        winner = payload["winner"]
        rows.append(
            {
                "line": line,
                "method": payload["method"],
                "rule": payload["rule"],
                "baseline_final": base["full"]["final_value"],
                "upgrade_final": winner["full"]["final_value"],
                "full_improvement": winner["full"]["final_value"] / base["full"]["final_value"] - 1.0,
                "baseline_win_rate": base["full"]["win_rate"],
                "upgrade_win_rate": winner["full"]["win_rate"],
                "baseline_avg_annual": base["full"]["avg_annual_return"],
                "upgrade_avg_annual": winner["full"]["avg_annual_return"],
                "baseline_max_drawdown": base["full"]["max_drawdown"],
                "upgrade_max_drawdown": winner["full"]["max_drawdown"],
                "holdout_baseline_final": base["holdout"]["final_value"],
                "holdout_upgrade_final": winner["holdout"]["final_value"],
                "holdout_improvement": winner["holdout"]["final_value"] / base["holdout"]["final_value"] - 1.0,
                "stability_count": payload["stability_count"],
            }
        )
    return pd.DataFrame(rows)


def report(metrics: pd.DataFrame, audits: list[dict]) -> str:
    lines = [
        "# C / S / D 滚动升级候选",
        "",
        "| 线 | 方法 | 原终值 | 新终值 | 提升 | 胜率 | 年均收益 | 最大回撤 | 2024—2026提升 | 稳定设置数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['line']} | {row['method']} | {row['baseline_final']:,.2f} | "
            f"{row['upgrade_final']:,.2f} | {row['full_improvement']:+.2%} | "
            f"{row['upgrade_win_rate']:.2%} | {row['upgrade_avg_annual']:.2%} | "
            f"{row['upgrade_max_drawdown']:.2%} | {row['holdout_improvement']:+.2%} | "
            f"{int(row['stability_count'])} |"
        )
    lines += [
        "",
        "## 审计",
        "",
        "| 线 | 有效交易 | 无效交易 | 重叠 |",
        "|---|---:|---:|---:|",
    ]
    for item in audits:
        lines.append(
            f"| {item['line']} | {item['valid']}/{item['trades']} | {item['invalid']} | {item['overlaps']} |"
        )
    lines += [
        "",
        "所有候选均保持原买点和仓位，仅改变退出覆盖层；信号与成交按当时可见行情生成。",
        "C 与 D 为年度扩展窗口，S 为截至 2023 年训练并从 2024 年冻结。",
        "由于研究过程中已经查看过 2024—2026，三者仍应先作为影子候选，而不是宣称全新盲测。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    core.configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    payloads = result_payloads()
    specs = {spec.key: spec for spec in core.SPECS}
    audits = []
    for line, source in SOURCES.items():
        frame = pd.read_csv(source, encoding="utf-8-sig", dtype={"symbol": str})
        frame["symbol"] = frame["symbol"].str.zfill(6)
        for column in ("entry_date", "exit_date_test", "control_signal_date"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        frame.to_csv(OUT / f"{line.lower()}_trades.csv", index=False, encoding="utf-8-sig")
        audits.append(audit_line(specs[line], frame))
    metrics = make_metrics(payloads)
    metrics.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    audit_payload = {"lines": audits}
    (OUT / "audit.json").write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "summary.json").write_text(
        json.dumps(payloads, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "REPORT.md").write_text(report(metrics, audits), encoding="utf-8")
    print(metrics.to_json(orient="records", force_ascii=False, indent=2))
    print(json.dumps(audit_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
