from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from baseline_audit import (
    audit_choice_causality,
    expected_training_years,
    single_account_overlap_count,
    trade_year_mismatches,
)

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "same_day_1445" / "release_v2"
EXPECTED = {"C": 38, "S": 44, "D": 1303, "R": 44}
INITIAL = 10_000.0
TOL = 1e-8

REQUIRED = [
    ROOT / "baseline.json",
    RELEASE / "README.md",
    RELEASE / "selected_summary.csv",
    RELEASE / "selected_choices.csv",
    RELEASE / "selected_trades_csr.csv",
    RELEASE / "d_annual_compact.csv",
    RELEASE / "d_cross_year_entry_audit.csv",
    RELEASE / "audit.json",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def num(value: Any) -> float:
    return float(str(value).strip())


def close(a: Any, b: Any, tol: float = TOL) -> bool:
    return math.isclose(num(a), num(b), rel_tol=tol, abs_tol=tol)


def fail(message: str) -> None:
    raise SystemExit(message)


def compound_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        return {"final": INITIAL, "trades": 0.0, "win": 0.0, "avg": 0.0, "positive_year_rate": 0.0, "max_dd": 0.0}
    ordered = sorted(rows, key=lambda row: (row["entry"], row.get("trade_id", "")))
    returns = [num(row["ret"]) for row in ordered]
    value = INITIAL
    equity = [value]
    for ret in returns:
        value *= 1.0 + ret
        equity.append(value)
    peak = equity[0]
    max_dd = 0.0
    for current in equity:
        peak = max(peak, current)
        max_dd = min(max_dd, current / peak - 1.0 if peak else 0.0)
    by_year: dict[int, list[float]] = defaultdict(list)
    for row in ordered:
        by_year[int(row["entry"][:4])].append(num(row["ret"]))
    positive = 0
    for values in by_year.values():
        annual = 1.0
        for ret in values:
            annual *= 1.0 + ret
        positive += annual - 1.0 > 0
    return {
        "final": value,
        "trades": float(len(ordered)),
        "win": sum(ret > 0 for ret in returns) / len(returns),
        "avg": sum(returns) / len(returns),
        "positive_year_rate": positive / len(by_year),
        "max_dd": max_dd,
    }


def verify_choice_evidence(choices: list[dict[str, str]], summary: dict[str, dict[str, str]]) -> None:
    causality = audit_choice_causality(choices)
    if not causality["ok"]:
        fail(f"causality evidence failed: {causality}")
    years_by_line: dict[str, list[int]] = defaultdict(list)
    for row in choices:
        years_by_line[row["line"]].append(int(row["year"]))
    for line in years_by_line:
        years_by_line[line] = sorted(set(years_by_line[line]))
    for row in choices:
        line = row["line"]
        year = int(row["year"])
        s = summary[line]
        expected = expected_training_years(
            years_by_line[line],
            year,
            int(float(s["window"])),
            int(float(s["min_years"])),
        )
        actual = [int(item) for item in row.get("train_years", "").split(";") if item]
        if actual != expected:
            fail(f"{line} {year} train_years mismatch: {actual} != {expected}")
        if not expected and row["policy_id"] != s["base_policy_id"]:
            fail(f"{line} {year} must use protected base before minimum history")


def verify_csr_trades(rows: list[dict[str, str]], summary: dict[str, dict[str, str]], audit: dict[str, Any]) -> None:
    by_line: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_line[row["line"]].append(row)
    if set(by_line) != {"C", "S", "R"}:
        fail(f"CSR ledger lines mismatch: {sorted(by_line)}")
    for line in ("C", "S", "R"):
        trades = by_line[line]
        if len(trades) != EXPECTED[line]:
            fail(f"{line} trade count mismatch: {len(trades)} != {EXPECTED[line]}")
        if trade_year_mismatches(trades):
            fail(f"{line} contains stored-year / effective-entry-year mismatches")
        missing = sum(1 for row in trades if not row.get("entry_close") or not row.get("exit_close"))
        bad_dates = sum(1 for row in trades if row["exit"] <= row["entry"])
        keys = [(row["symbol"], row["entry"], row["exit"]) for row in trades]
        duplicates = len(keys) - len(set(keys))
        overlap = single_account_overlap_count(trades)
        stored = audit.get("lines", {}).get(line, {})
        checks = {
            "trade_rows": len(trades),
            "missing_prices": missing,
            "bad_dates": bad_dates,
            "duplicate_trade_keys": duplicates,
            "year_entry_mismatch": 0,
            "overlap_count": overlap,
        }
        for key, expected in checks.items():
            if stored.get(key) != expected:
                fail(f"{line} audit {key} mismatch: {stored.get(key)} != {expected}")
        full = compound_metrics(trades)
        holdout = compound_metrics([row for row in trades if int(row["entry"][:4]) >= 2024])
        field_map = {
            "release_final": "final",
            "release_trades": "trades",
            "release_win": "win",
            "release_avg": "avg",
            "release_positive_year_rate": "positive_year_rate",
            "release_max_dd": "max_dd",
        }
        holdout_map = {
            "holdout_final": "final",
            "holdout_trades": "trades",
            "holdout_win": "win",
            "holdout_avg": "avg",
            "holdout_positive_year_rate": "positive_year_rate",
            "holdout_max_dd": "max_dd",
        }
        for csv_key, metric_key in field_map.items():
            if not close(summary[line][csv_key], full[metric_key], tol=1e-7):
                fail(f"{line} {csv_key} mismatch")
        for csv_key, metric_key in holdout_map.items():
            if not close(summary[line][csv_key], holdout[metric_key], tol=1e-7):
                fail(f"{line} {csv_key} mismatch")


def verify_d_annual(
    annual_rows: list[dict[str, str]],
    choices: list[dict[str, str]],
    summary: dict[str, dict[str, str]],
    cross_year_rows: list[dict[str, str]],
    audit: dict[str, Any],
) -> None:
    d_choices = {int(row["year"]): row for row in choices if row["line"] == "D"}
    annual = {int(row["year"]): row for row in annual_rows}
    if set(annual) != set(d_choices):
        fail(f"D annual years mismatch: {sorted(annual)} != {sorted(d_choices)}")

    total_trades = 0
    total_wins = 0.0
    total_sum_return = 0.0
    holdout_trades = 0
    holdout_wins = 0.0
    holdout_sum_return = 0.0
    for year in sorted(annual):
        row = annual[year]
        choice = d_choices[year]
        trades = int(row["trades"])
        win_rate = num(row["win_rate"])
        sum_return = num(row["sum_return"])
        avg_return = num(row["avg_return"])
        if trades != int(float(choice["test_trades"])):
            fail(f"D {year} annual trades do not match selected_choices.csv")
        if not close(win_rate, choice["test_win"], tol=1e-7):
            fail(f"D {year} annual win_rate does not match selected_choices.csv")
        expected_sum = num(choice["test_final"]) / INITIAL - 1.0
        if not close(sum_return, expected_sum, tol=1e-7):
            fail(f"D {year} annual sum_return does not match selected_choices.csv")
        expected_avg = sum_return / trades if trades else 0.0
        if not close(avg_return, expected_avg, tol=1e-7):
            fail(f"D {year} annual avg_return mismatch")
        total_trades += trades
        total_wins += trades * win_rate
        total_sum_return += sum_return
        if year >= 2024:
            holdout_trades += trades
            holdout_wins += trades * win_rate
            holdout_sum_return += sum_return

    d_summary = summary["D"]
    aggregate_checks = {
        "release_trades": float(total_trades),
        "release_final": INITIAL * (1.0 + total_sum_return),
        "release_win": total_wins / total_trades,
        "release_avg": total_sum_return / total_trades,
        "holdout_trades": float(holdout_trades),
        "holdout_final": INITIAL * (1.0 + holdout_sum_return),
        "holdout_win": holdout_wins / holdout_trades,
        "holdout_avg": holdout_sum_return / holdout_trades,
    }
    for key, expected in aggregate_checks.items():
        if not close(d_summary[key], expected, tol=1e-7):
            fail(f"D aggregate {key} mismatch")

    if total_trades != EXPECTED["D"]:
        fail(f"D total trades mismatch: {total_trades} != {EXPECTED['D']}")

    # The six known boundary shifts are kept as compact per-trade evidence so
    # the effective entry-year convention can be checked without committing
    # the full 1303-row D research ledger.
    if len(cross_year_rows) != 6:
        fail(f"D cross-year audit row count mismatch: {len(cross_year_rows)} != 6")
    for row in cross_year_rows:
        entry_year = int(row["entry"][:4])
        source_year = int(row["old_entry"][:4])
        if int(row["year"]) != entry_year or int(row["entry_year"]) != entry_year:
            fail(f"{row['trade_id']} effective entry-year mismatch")
        if int(row["source_year"]) != source_year:
            fail(f"{row['trade_id']} source-year mismatch")
        if entry_year == source_year:
            fail(f"{row['trade_id']} is not actually a cross-year shift")

    stored = audit.get("lines", {}).get("D", {})
    if stored.get("trade_rows") != total_trades:
        fail("D audit trade_rows does not match annual compact total")
    if stored.get("year_entry_mismatch") != 0:
        fail("D audit reports an effective entry-year mismatch")
    if stored.get("all_year_choices_are_causal") is not True:
        fail("D audit causality flag is not true")


def verify_baseline_json(summary_rows: list[dict[str, str]]) -> None:
    baseline = json.loads((ROOT / "baseline.json").read_text(encoding="utf-8-sig"))
    if baseline.get("version") != "2026-08-21-release-v2":
        fail("baseline.json version mismatch")
    summary = {row["line"]: row for row in summary_rows}
    for line in EXPECTED:
        metric = baseline.get("metrics", {}).get(line, {})
        checks = {
            "final_equity": "release_final",
            "trades": "release_trades",
            "win_rate": "release_win",
            "avg_trade_return": "release_avg",
            "max_drawdown": "release_max_dd",
            "holdout_2024_win_rate": "holdout_win",
        }
        for json_key, csv_key in checks.items():
            if json_key == "trades":
                if int(metric[json_key]) != int(float(summary[line][csv_key])):
                    fail(f"baseline.json {line} {json_key} mismatch")
            elif not close(metric[json_key], summary[line][csv_key], tol=5e-4):
                fail(f"baseline.json {line} {json_key} mismatch")


def main() -> int:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.exists()]
    if missing:
        fail("MISSING: " + ", ".join(missing))

    summary_rows = read_csv(RELEASE / "selected_summary.csv")
    summary = {row["line"]: row for row in summary_rows}
    if set(summary) != set(EXPECTED):
        fail(f"summary strategy lines mismatch: {sorted(summary)}")
    choices = read_csv(RELEASE / "selected_choices.csv")
    csr_rows = read_csv(RELEASE / "selected_trades_csr.csv")
    annual_rows = read_csv(RELEASE / "d_annual_compact.csv")
    cross_year_rows = read_csv(RELEASE / "d_cross_year_entry_audit.csv")
    audit = json.loads((RELEASE / "audit.json").read_text(encoding="utf-8-sig"))

    verify_choice_evidence(choices, summary)
    verify_csr_trades(csr_rows, summary, audit)
    verify_d_annual(annual_rows, choices, summary, cross_year_rows, audit)
    verify_baseline_json(summary_rows)

    if audit.get("ok") is not True:
        fail("release audit is not OK")

    print("CURRENT BASELINE OK")
    print("- C/S/R: 126 frozen trades recomputed from per-trade ledger")
    print("- D: 1303 trades reconciled from annual compact + yearly choices")
    print("- causal training windows: exact strictly-prior evidence verified")
    print("- D cross-year entry convention: 6 boundary shifts verified")
    print("- baseline.json, selected_summary.csv and audit.json are consistent")
    print("NOTE: full D per-trade drawdown reconstruction still requires the archived source package")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
