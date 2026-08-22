from __future__ import annotations

from datetime import date
from typing import Any


def expected_training_years(years: list[int], year: int, window: int, min_years: int) -> list[int]:
    """Return the exact strictly-prior training years for one test year."""
    prior = sorted({int(item) for item in years if int(item) < int(year)})
    if len(prior) < int(min_years):
        return []
    return prior[-int(window):] if int(window) else prior


def parse_train_years(value: Any) -> list[int]:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [int(part) for part in text.split(";") if part.strip()]


def audit_choice_causality(choices: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    evidence_errors: list[dict[str, Any]] = []
    for item in choices:
        line = str(item.get("line", ""))
        year = int(item["year"])
        train_years = parse_train_years(item.get("train_years"))
        try:
            stored_count = int(float(str(item.get("train_year_count", len(train_years))) or 0))
        except ValueError:
            stored_count = -1
        if stored_count != len(train_years):
            evidence_errors.append({
                "line": line,
                "year": year,
                "reason": "train_year_count_mismatch",
                "stored_count": stored_count,
                "parsed_count": len(train_years),
            })
        for train_year in train_years:
            if train_year >= year:
                violations.append({"line": line, "year": year, "train_year": train_year})
        flag = str(item.get("causal_train_window", "")).strip().lower()
        if flag not in {"true", "1"}:
            evidence_errors.append({
                "line": line,
                "year": year,
                "reason": "causal_train_window_flag_false",
            })
    return {
        "ok": not violations and not evidence_errors,
        "violations": violations,
        "evidence_errors": evidence_errors,
    }


def trade_year_mismatches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for row in rows:
        stored_year = int(float(str(row["year"])))
        entry_year = date.fromisoformat(str(row["entry"])[:10]).year
        if stored_year != entry_year:
            mismatches.append({
                "line": str(row.get("line", "")),
                "trade_id": str(row.get("trade_id", "")),
                "stored_year": stored_year,
                "entry_year": entry_year,
            })
    return mismatches


def single_account_overlap_count(rows: list[dict[str, Any]]) -> int:
    ordered = sorted(rows, key=lambda row: (str(row["entry"]), str(row["exit"]), str(row.get("trade_id", ""))))
    return sum(1 for left, right in zip(ordered, ordered[1:]) if str(right["entry"]) < str(left["exit"]))


def release_is_ok(line_audits: dict[str, dict[str, Any]]) -> bool:
    for line, item in line_audits.items():
        if int(item.get("missing_prices", -1)) != 0:
            return False
        if int(item.get("bad_dates", -1)) != 0:
            return False
        if int(item.get("duplicate_trade_keys", -1)) != 0:
            return False
        if int(item.get("year_entry_mismatch", -1)) != 0:
            return False
        if item.get("all_year_choices_are_causal") is not True:
            return False
        if line != "D" and item.get("overlap_count") != 0:
            return False
    return True
