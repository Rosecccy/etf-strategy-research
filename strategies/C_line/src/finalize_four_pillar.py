from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

import four_pillar_rolling_test as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "four_pillar" / "accepted"


def annual_rows(line: str, frame: pd.DataFrame) -> pd.DataFrame:
    _, detail, annual = core.account.simulate_account(frame)
    annual["line"] = line
    return annual[["line", "year", "account_return"]]


def stats_row(line: str, status: str, method: str, frame: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    stats = core.exact_stats(frame)
    base = core.exact_stats(baseline)
    holdout = core.period_stats(frame, start=2024)
    base_holdout = core.period_stats(baseline, start=2024)
    return {
        "line": line,
        "status": status,
        "method": method,
        "baseline_final": base["final_value"],
        "final_value": stats["final_value"],
        "improvement": stats["final_value"] / base["final_value"] - 1.0,
        "win_rate": stats["win_rate"],
        "avg_annual_return": stats["avg_annual_return"],
        "max_drawdown": stats["max_drawdown"],
        "trades": stats["trades"],
        "trigger_retention": stats["trades"] / base["trades"],
        "holdout_ratio": holdout["final_value"] / base_holdout["final_value"],
        "holdout_win": holdout["win_rate"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(core.FEATURES)
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel["date"])

    baselines = {line: core.load_trades(line) for line in ("C", "S", "D")}
    c_path = pd.read_csv(ROOT / "fit" / "four_pillar" / "guarded" / "c_best_trades.csv", dtype={"symbol": str})
    for column in ("entry_date", "exit_date_test"):
        c_path[column] = pd.to_datetime(c_path[column])

    s_base = baselines["S"]
    s_frame = core.attach_shift_prices(
        core.attach_features(s_base, panel), core.load_prices(set(s_base["symbol"]))
    )
    s_path, s_selected = core.apply_path(s_frame, "all_four", "all")
    s_selected.to_csv(OUT / "s_selected_by_year.csv", index=False, encoding="utf-8-sig")

    d_path = pd.read_csv(ROOT / "fit" / "four_pillar" / "guarded" / "d_best_trades.csv", dtype={"symbol": str})
    for column in ("entry_date", "exit_date_test"):
        d_path[column] = pd.to_datetime(d_path[column])

    paths = {"C": c_path, "S": s_path, "D": d_path}
    for line, frame in paths.items():
        frame["factor_reversal_score"] = pd.concat(
            [
                pd.to_numeric(frame["pillar_volume_reversal"], errors="coerce").fillna(0.5),
                1.0 - pd.to_numeric(frame["pillar_turnover"], errors="coerce").fillna(0.5),
                pd.to_numeric(frame["pillar_valuation"], errors="coerce").fillna(0.5),
                pd.to_numeric(frame["pillar_chip_reversal"], errors="coerce").fillna(0.5),
            ],
            axis=1,
        ).mean(axis=1)
        frame["factor_momentum_score"] = pd.concat(
            [
                pd.to_numeric(frame["pillar_volume_momentum"], errors="coerce").fillna(0.5),
                pd.to_numeric(frame["pillar_turnover"], errors="coerce").fillna(0.5),
                pd.to_numeric(frame["pillar_valuation"], errors="coerce").fillna(0.5),
                pd.to_numeric(frame["pillar_chip_momentum"], errors="coerce").fillna(0.5),
            ],
            axis=1,
        ).mean(axis=1)
        frame.to_csv(OUT / f"{line.lower()}_trades.csv", index=False, encoding="utf-8-sig")

    rows = [
        stats_row("C", "accepted", "四因子胜率保护；3年滚动；仅调整退出时点", c_path, baselines["C"]),
        stats_row("S", "accepted", "四因子扩展窗口；全部过去历史滚动；仅调整退出时点", s_path, baselines["S"]),
        stats_row("D", "observer", "四因子评分已接入；交易改动未通过，保持原规则", d_path, baselines["D"]),
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    pd.concat([annual_rows(line, frame) for line, frame in paths.items()], ignore_index=True).to_csv(
        OUT / "annual.csv", index=False, encoding="utf-8-sig"
    )
    payload = {
        "as_of": "2026-07-31",
        "execution": "factor values are taken from the last trading day strictly before entry",
        "valuation": "historical broad-market PE/PB percentile; no current-value backfill",
        "chip": "causal daily-bar turnover-decay cost-distribution proxy; not proprietary broker chip data",
        "decision": summary.to_dict(orient="records"),
    }
    (OUT / "decision.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(ROOT / "factors" / "panel_summary.json", OUT / "panel_summary.json")
    print(summary.drop(columns=["method"]).to_string(index=False))


if __name__ == "__main__":
    main()
