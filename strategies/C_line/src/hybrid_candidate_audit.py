from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OUT = ROOT / "fit" / "hybrid_test"
C_FORMAL = ROOT / "fit" / "fallback" / "final_trades.csv"
S_FORMAL = WORKSPACE / "S" / "fit" / "selector" / "final_trades.csv"
CANDIDATE = OUT / "winner_trades.csv"
HOLDOUT_YEARS = {2024, 2025, 2026}


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    frame["exit_date"] = pd.to_datetime(frame["exit_date"], errors="coerce")
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    if "decision_date" in frame:
        frame["decision_date"] = pd.to_datetime(
            frame["decision_date"], errors="coerce", format="mixed"
        )
    return frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def compound(frame: pd.DataFrame, years: set[int] | None = None) -> float:
    local = frame
    if years is not None:
        local = local[local["entry_date"].dt.year.isin(years)]
    return float(1000.0 * np.prod(1.0 + local["ret"].dropna()))


def yearly(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    local = frame.assign(year=frame["entry_date"].dt.year)
    result = (
        local.groupby("year")["ret"]
        .agg(
            trades="size",
            wins=lambda values: int((values > 0).sum()),
            win_rate=lambda values: float((values > 0).mean()),
            avg_return="mean",
            annual_return=lambda values: float(
                np.prod(1.0 + values.dropna()) - 1.0
            ),
        )
        .reset_index()
    )
    return result.rename(
        columns={
            column: f"{label}_{column}"
            for column in result.columns
            if column != "year"
        }
    )


def fallback_causality(candidate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fallback = candidate[candidate["family"].isin(["C", "S"])].copy()
    for _, trade in fallback.iterrows():
        family = str(trade["family"])
        preferred = ROOT if family == "C" else WORKSPACE / "S"
        path = preferred / "raw" / "etf" / f"{trade['symbol']}.csv"
        prices = pd.read_csv(path, encoding="utf-8-sig")
        dates = pd.DatetimeIndex(
            pd.to_datetime(prices["date"], errors="coerce")
            .dropna()
            .drop_duplicates()
            .sort_values()
        )
        decision = pd.Timestamp(trade["decision_date"])
        entry = pd.Timestamp(trade["entry_date"])
        next_pos = int(dates.searchsorted(decision, side="right"))
        expected_entry = (
            pd.Timestamp(dates[next_pos]) if next_pos < len(dates) else pd.NaT
        )
        rows.append(
            {
                "family": family,
                "source": trade["source"],
                "symbol": trade["symbol"],
                "decision_date": decision,
                "entry_date": entry,
                "expected_next_trade_date": expected_entry,
                "decision_before_entry": bool(decision < entry),
                "is_t_plus_1_close": bool(entry == expected_entry),
            }
        )
    return pd.DataFrame(rows)


def overlap_audit(candidate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    previous_exit = pd.Timestamp.min
    for _, trade in candidate.sort_values(["entry_date", "symbol"]).iterrows():
        entry = pd.Timestamp(trade["entry_date"])
        rows.append(
            {
                "symbol": trade["symbol"],
                "entry_date": entry,
                "exit_date": trade["exit_date"],
                "previous_exit_date": (
                    previous_exit if previous_exit != pd.Timestamp.min else pd.NaT
                ),
                "overlap": bool(entry < previous_exit),
            }
        )
        previous_exit = max(previous_exit, pd.Timestamp(trade["exit_date"]))
    return pd.DataFrame(rows)


def leave_one_year_out(
    candidate: pd.DataFrame,
    c_formal: pd.DataFrame,
    s_formal: pd.DataFrame,
) -> pd.DataFrame:
    all_years = sorted(candidate["entry_date"].dt.year.unique())
    rows = []
    for omitted in all_years:
        years = set(all_years) - {int(omitted)}
        candidate_value = compound(candidate, years)
        c_value = compound(c_formal, years)
        s_value = compound(s_formal, years)
        best_baseline = max(c_value, s_value)
        rows.append(
            {
                "omitted_year": int(omitted),
                "candidate_final_1000": candidate_value,
                "c_final_1000": c_value,
                "s_final_1000": s_value,
                "gain_vs_best_baseline": candidate_value / best_baseline - 1.0,
                "beats_both": bool(candidate_value > best_baseline),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidate = load(CANDIDATE)
    c_formal = load(C_FORMAL)
    s_formal = load(S_FORMAL)
    main_label = str(c_formal["source"].value_counts().idxmax())

    candidate_main = candidate[candidate["source"].eq(main_label)]
    c_main = c_formal[c_formal["source"].eq(main_label)]
    keys = ["symbol", "entry_date", "exit_date"]
    main_join = c_main[keys].merge(
        candidate_main[keys],
        on=keys,
        how="outer",
        indicator=True,
    )
    missing_main = int((main_join["_merge"] == "left_only").sum())
    extra_main = int((main_join["_merge"] == "right_only").sum())

    origins = pd.concat(
        [
            c_formal.assign(origin_line="C"),
            s_formal.assign(origin_line="S"),
        ],
        ignore_index=True,
        sort=False,
    )
    origin_keys = ["source", "symbol", "entry_date", "exit_date"]
    origin_set = set(map(tuple, origins[origin_keys].itertuples(index=False)))
    candidate["has_formal_origin"] = [
        tuple(row) in origin_set
        for row in candidate[origin_keys].itertuples(index=False, name=None)
    ]
    missing_origin = int((~candidate["has_formal_origin"]).sum())

    causal = fallback_causality(candidate)
    overlaps = overlap_audit(candidate)
    loo = leave_one_year_out(candidate, c_formal, s_formal)

    annual = yearly(candidate, "candidate")
    annual = annual.merge(yearly(c_formal, "c"), on="year", how="outer")
    annual = annual.merge(yearly(s_formal, "s"), on="year", how="outer")
    annual["candidate_vs_best"] = annual["candidate_annual_return"] - annual[
        ["c_annual_return", "s_annual_return"]
    ].max(axis=1)

    holdout_candidate = compound(candidate, HOLDOUT_YEARS)
    holdout_c = compound(c_formal, HOLDOUT_YEARS)
    holdout_s = compound(s_formal, HOLDOUT_YEARS)
    holdout_gain = holdout_candidate / max(holdout_c, holdout_s) - 1.0

    checks = {
        "all_c_main_trades_preserved": missing_main == 0 and extra_main == 0,
        "all_trades_have_formal_origin": missing_origin == 0,
        "no_account_overlap": int(overlaps["overlap"].sum()) == 0,
        "all_fallback_decisions_before_entry": bool(
            causal["decision_before_entry"].all()
        ),
        "all_fallback_entries_t_plus_1": bool(causal["is_t_plus_1_close"].all()),
        "all_leave_one_year_out_beats_both": bool(loo["beats_both"].all()),
        "holdout_gain_above_5pct": holdout_gain > 0.05,
        "holdout_has_at_least_10_trades": int(
            candidate["entry_date"].dt.year.isin(HOLDOUT_YEARS).sum()
        )
        >= 10,
    }
    passed = bool(all(checks.values()))

    causal.to_csv(OUT / "causality_audit.csv", index=False, encoding="utf-8-sig")
    overlaps.to_csv(OUT / "overlap_audit.csv", index=False, encoding="utf-8-sig")
    loo.to_csv(
        OUT / "leave_one_year_out.csv", index=False, encoding="utf-8-sig"
    )
    annual.to_csv(OUT / "annual_compare.csv", index=False, encoding="utf-8-sig")
    candidate.to_csv(
        OUT / "formal_candidate_trades.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "candidate": "C main and exit + causal union of C/S fallback",
        "checks": checks,
        "passed": passed,
        "missing_main_trades": missing_main,
        "extra_main_trades": extra_main,
        "missing_formal_origins": missing_origin,
        "candidate_trades": int(len(candidate)),
        "candidate_win_rate": float((candidate["ret"] > 0).mean()),
        "candidate_gross_final_1000": compound(candidate),
        "c_gross_final_1000": compound(c_formal),
        "s_gross_final_1000": compound(s_formal),
        "holdout_candidate_final_1000": holdout_candidate,
        "holdout_best_baseline_final_1000": max(holdout_c, holdout_s),
        "holdout_gain_vs_best": holdout_gain,
        "minimum_leave_one_year_out_gain": float(
            loo["gain_vs_best_baseline"].min()
        ),
    }
    (OUT / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
