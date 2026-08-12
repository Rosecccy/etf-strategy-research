from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import fear_greed_oos as fg
import fear_greed_weight_sweep as sweep
from control_sizing_test import account


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "gate_portfolio_test"
GATE_OUT = ROOT / "out" / "ma120_gate_test"
DEV_YEARS = (2021, 2022, 2023)
HOLDOUT_YEARS = (2025, 2026)
QVIX_WEIGHT = 0.20
ENTRY_THRESHOLD = 45.0
GREED_EXIT = 70.0
MAX_HOLD = 90


def selection_score(summary: dict[str, float | int]) -> float:
    return (
        math.log(max(float(summary["final_value"]), 1.0) / fg.INITIAL_CAPITAL)
        + 0.25 * float(np.nan_to_num(summary["win_rate"], nan=0.0))
        + 0.70 * float(summary["max_drawdown"])
    )


def prepare_gate_candidates(
    trades: pd.DataFrame,
    raw: pd.DataFrame,
    sentiment: pd.DataFrame,
    symbol_fear: pd.DataFrame,
) -> pd.DataFrame:
    local = fg.rebuild_trade_prices(trades, raw)
    local = fg.attach_sentiment(local, sentiment, symbol_fear)
    local["fear_weighted"] = sweep.composite(
        local["fear_core"], local["fear_qvix"], QVIX_WEIGHT
    )
    local["buy_quality"] = (
        0.5 * local["fear_weighted"] + 0.5 * local["symbol_fear"]
    )
    local_sentiment = sentiment.copy()
    local_sentiment["fear_enhanced"] = sweep.composite(
        local_sentiment["fear_core"],
        local_sentiment["fear_qvix"],
        QVIX_WEIGHT,
    )
    local_sentiment["greed_enhanced"] = 100 - local_sentiment["fear_enhanced"]
    adjusted = fg.apply_exit_rule(
        local,
        fg.ExitRule("formal_fg1_exit", MAX_HOLD, GREED_EXIT),
        raw,
        local_sentiment,
    )
    selected = adjusted[
        adjusted["fear_weighted"].ge(ENTRY_THRESHOLD)
    ].copy()
    selected["overlay"] = "fear_greed"
    return selected


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    _, sentiment, adjusted_map, raw, pool, _ = sweep.generate()
    symbol_fear = fg.build_symbol_fear(raw)

    formal = adjusted_map[(QVIX_WEIGHT, MAX_HOLD, GREED_EXIT)].copy()
    formal = formal[formal["fear_weighted"].ge(ENTRY_THRESHOLD)].copy()
    formal["overlay"] = "fear_greed"

    candidate_sources = {
        "formal_baseline": formal,
    }
    for mode in ("win", "efficiency"):
        trades = pd.read_csv(
            GATE_OUT / f"rolling_{mode}_trades.csv",
            dtype={"symbol": str},
            encoding="utf-8-sig",
        )
        candidate_sources[f"rolling_{mode}"] = prepare_gate_candidates(
            trades, raw, sentiment, symbol_fear
        )

    rows: list[dict[str, object]] = []
    logs: dict[tuple[str, str], pd.DataFrame] = {}
    summaries: dict[tuple[str, str], dict[str, float | int]] = {}
    for candidate, source in candidate_sources.items():
        for stage, years in (
            ("development", DEV_YEARS),
            ("holdout", HOLDOUT_YEARS),
            ("all", None),
        ):
            local = (
                source
                if years is None
                else source[source["test_year"].isin(years)].copy()
            )
            log, summary = account(local, raw, pool, "full")
            logs[(candidate, stage)] = log
            summaries[(candidate, stage)] = summary
            rows.append(
                {
                    "candidate": candidate,
                    "stage": stage,
                    "selection_score": selection_score(summary),
                    **summary,
                }
            )
        print(f"[D gate portfolio] {candidate}")

    results = pd.DataFrame(rows)
    development = results[results["stage"].eq("development")].sort_values(
        ["selection_score", "final_value"], ascending=False
    )
    winner = str(development.iloc[0]["candidate"])
    baseline_holdout = summaries[("formal_baseline", "holdout")]
    winner_holdout = summaries[(winner, "holdout")]
    pass_count = int(
        sum(
            summaries[(candidate, "holdout")]["final_value"]
            >= baseline_holdout["final_value"]
            and summaries[(candidate, "holdout")]["win_rate"]
            >= baseline_holdout["win_rate"] - 0.02
            and summaries[(candidate, "holdout")]["max_drawdown"]
            >= baseline_holdout["max_drawdown"] - 0.01
            for candidate in candidate_sources
        )
    )
    promoted = bool(
        winner != "formal_baseline"
        and winner_holdout["final_value"] > baseline_holdout["final_value"]
        and winner_holdout["win_rate"] >= baseline_holdout["win_rate"]
        and winner_holdout["max_drawdown"]
        >= baseline_holdout["max_drawdown"]
        and pass_count >= 2
    )

    results.to_csv(OUT / "variants.csv", index=False, encoding="utf-8-sig")
    logs[(winner, "all")].to_csv(
        OUT / "winner_trades.csv", index=False, encoding="utf-8-sig"
    )
    logs[(winner, "holdout")].to_csv(
        OUT / "winner_holdout_trades.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "control_variable": "ma120_entry_gate_only",
        "fixed_formal_rules": {
            "qvix_weight": QVIX_WEIGHT,
            "entry_threshold": ENTRY_THRESHOLD,
            "greed_exit": GREED_EXIT,
            "max_hold": MAX_HOLD,
            "execution": "T+1 close",
            "portfolio": "one position, full allocation, 100-share lots",
            "fee": "0.03%, minimum CNY 5 per side",
        },
        "development_selected_candidate": winner,
        "baseline_holdout": baseline_holdout,
        "winner_holdout": winner_holdout,
        "holdout_candidates_passing_floor": pass_count,
        "promoted": promoted,
        "formal_config_changed": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    report = (
        "# D 线 MA120 门槛实盘账户复核\n\n"
        f"- 开发期选择：{winner}\n"
        f"- 正式基线留出期终值：{baseline_holdout['final_value']:.2f}\n"
        f"- 候选留出期终值：{winner_holdout['final_value']:.2f}\n"
        f"- 正式基线留出期胜率：{baseline_holdout['win_rate']:.2%}\n"
        f"- 候选留出期胜率：{winner_holdout['win_rate']:.2%}\n"
        f"- 候选留出期最大回撤：{winner_holdout['max_drawdown']:.2%}\n"
        f"- 邻域通过数量：{pass_count}\n"
        f"- 是否升级：{'是' if promoted else '否'}\n\n"
        "本表把 MA120 门槛候选重新放回 FG1 单仓、整百股和真实手续费账户，"
        "因此可直接与当前 D 正式影子版比较。"
    )
    (OUT / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
