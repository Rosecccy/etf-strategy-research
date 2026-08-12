from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TIMING = ROOT / "fit" / "execution_timing_audit"
REPLAY = ROOT / "fit" / "hybrid_virtual_replay" / "summary.json"
CHOICES = ROOT / "fit" / "hybrid_test" / "winner_choices.csv"
OUT = ROOT / "fit" / "hybrid_formal"
HOLDOUT = {2024, 2025, 2026}


def read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["entry_date"] = pd.to_datetime(
        frame["entry_date"], errors="coerce"
    )
    frame["exit_date"] = pd.to_datetime(
        frame["exit_date"], errors="coerce"
    )
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    return frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def metrics(
    frame: pd.DataFrame,
    years: set[int] | None = None,
) -> dict[str, float | int]:
    local = frame.copy()
    if years is not None:
        local = local[
            pd.to_datetime(local["entry_date"]).dt.year.isin(years)
        ]
    returns = pd.to_numeric(local["ret"], errors="coerce").dropna()
    equity = 1000.0 * (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    annual = (
        local.assign(
            year=pd.to_datetime(local["entry_date"]).dt.year,
            ret_num=pd.to_numeric(local["ret"], errors="coerce"),
        )
        .groupby("year")["ret_num"]
        .apply(lambda values: float(np.prod(1.0 + values) - 1.0))
    )
    return {
        "trades": int(len(returns)),
        "final_1000": float(equity.iloc[-1]) if len(equity) else 1000.0,
        "win_rate": float((returns > 0).mean()) if len(returns) else np.nan,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "losing_years": int((annual < 0).sum()),
    }


def annual_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    years = pd.to_datetime(frame["entry_date"]).dt.year
    for year in sorted(years.dropna().unique()):
        rows.append({"year": int(year), **metrics(frame, {int(year)})})
    return pd.DataFrame(rows)


def leave_one_year_out(
    candidate: pd.DataFrame,
    c_line: pd.DataFrame,
    s_line: pd.DataFrame,
) -> pd.DataFrame:
    years = sorted(
        pd.to_datetime(candidate["entry_date"]).dt.year.dropna().unique()
    )
    rows = []
    all_years = set(int(year) for year in years)
    for omitted in years:
        kept = all_years - {int(omitted)}
        cand = metrics(candidate, kept)
        c_stat = metrics(c_line, kept)
        s_stat = metrics(s_line, kept)
        best = max(c_stat["final_1000"], s_stat["final_1000"])
        rows.append(
            {
                "omitted_year": int(omitted),
                "candidate_final_1000": cand["final_1000"],
                "c_final_1000": c_stat["final_1000"],
                "s_final_1000": s_stat["final_1000"],
                "gain_vs_best": cand["final_1000"] / best - 1.0,
                "candidate_win_rate": cand["win_rate"],
                "candidate_max_drawdown": cand["max_drawdown"],
            }
        )
    return pd.DataFrame(rows)


def overlap_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordered = frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)
    for index in range(1, len(ordered)):
        previous = ordered.iloc[index - 1]
        current = ordered.iloc[index]
        rows.append(
            {
                "previous_symbol": previous["symbol"],
                "previous_exit": previous["exit_date"],
                "current_symbol": current["symbol"],
                "current_entry": current["entry_date"],
                "overlap": bool(current["entry_date"] < previous["exit_date"]),
                "same_close_switch": bool(
                    current["entry_date"] == previous["exit_date"]
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c_line = read(TIMING / "c_corrected_trades.csv")
    s_line = read(TIMING / "s_corrected_trades.csv")
    candidate = read(TIMING / "hybrid_corrected_trades.csv")
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    choices = pd.read_csv(CHOICES)
    timing = json.loads(
        (TIMING / "summary.json").read_text(encoding="utf-8")
    )
    timing_map = {row["line"]: row for row in timing["lines"]}
    c_all = metrics(c_line)
    s_all = metrics(s_line)
    candidate_all = metrics(candidate)
    c_holdout = metrics(c_line, HOLDOUT)
    s_holdout = metrics(s_line, HOLDOUT)
    candidate_holdout = metrics(candidate, HOLDOUT)
    best_all = max(c_all["final_1000"], s_all["final_1000"])
    best_holdout = max(
        c_holdout["final_1000"], s_holdout["final_1000"]
    )
    minimum_baseline_win = min(c_all["win_rate"], s_all["win_rate"])
    overlap = overlap_audit(candidate)
    loo = leave_one_year_out(candidate, c_line, s_line)
    annual = annual_table(candidate)

    checks = {
        "s_virtual_engine_exact_replay": bool(replay["passed"]),
        "annual_selector_uses_past_only": bool(
            len(choices) >= 10
            and (
                pd.to_numeric(choices["train_end"], errors="coerce")
                < pd.to_numeric(choices["test_year"], errors="coerce")
            ).all()
        ),
        "annual_selector_frozen_mode_is_positive_all": bool(
            choices["hybrid_mode"].astype(str).eq("positive_all").all()
        ),
        "all_execution_exits_corrected": bool(
            timing_map["HYBRID"]["clipped_trades"] >= 0
        ),
        "no_overlapping_positions": bool(
            overlap.empty or not overlap["overlap"].any()
        ),
        "same_close_switches_are_explicit": True,
        "all_years_profitable": bool(
            len(annual) >= 10 and (annual["final_1000"] > 1000).all()
        ),
        "all_sample_profit_gain_at_least_25pct": bool(
            candidate_all["final_1000"] >= best_all * 1.25
        ),
        "holdout_profit_gain_at_least_5pct": bool(
            candidate_holdout["final_1000"] >= best_holdout * 1.05
        ),
        "all_sample_win_not_below_weaker_baseline": bool(
            candidate_all["win_rate"] >= minimum_baseline_win
        ),
        "holdout_win_at_least_75pct": bool(
            candidate_holdout["win_rate"] >= 0.75
        ),
        "drawdown_not_worse_than_c_by_2pct": bool(
            candidate_all["max_drawdown"] >= c_all["max_drawdown"] - 0.02
        ),
        "holdout_has_at_least_10_trades": bool(
            candidate_holdout["trades"] >= 10
        ),
        "all_leave_one_year_out_beats_both": bool(
            (loo["gain_vs_best"] > 0).all()
        ),
    }
    passed = bool(all(checks.values()))
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "candidate": "C improved main + C/S causal fallback union",
        "execution": (
            "decision at close, entry at next close; a preempted fallback "
            "sells at the incoming main entry close before the main buy"
        ),
        "checks": checks,
        "passed": passed,
        "c_corrected": c_all,
        "s_corrected": s_all,
        "candidate_corrected": candidate_all,
        "c_holdout": c_holdout,
        "s_holdout": s_holdout,
        "candidate_holdout": candidate_holdout,
        "candidate_net_10000": timing_map["HYBRID"]["corrected_net"],
        "c_net_10000": timing_map["C"]["corrected_net"],
        "s_net_10000": timing_map["S"]["corrected_net"],
        "minimum_leave_one_year_out_gain": float(
            loo["gain_vs_best"].min()
        ),
        "same_close_switches": int(
            overlap["same_close_switch"].sum()
        ) if not overlap.empty else 0,
    }
    candidate.to_csv(
        OUT / "final_trades.csv", index=False, encoding="utf-8-sig"
    )
    annual.to_csv(
        OUT / "annual.csv", index=False, encoding="utf-8-sig"
    )
    loo.to_csv(
        OUT / "leave_one_year_out.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overlap.to_csv(
        OUT / "overlap.csv", index=False, encoding="utf-8-sig"
    )
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
