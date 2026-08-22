from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from baseline_audit import (  # noqa: E402
    audit_choice_causality,
    expected_training_years,
    release_is_ok,
    trade_year_mismatches,
)


def test_expected_training_years_are_strictly_prior():
    years = list(range(2014, 2027))
    assert expected_training_years(years, 2016, window=3, min_years=3) == []
    assert expected_training_years(years, 2017, window=3, min_years=3) == [2014, 2015, 2016]
    assert expected_training_years(years, 2024, window=3, min_years=3) == [2021, 2022, 2023]


def test_causality_audit_rejects_future_training_year():
    choices = [
        {"line": "C", "year": "2024", "train_years": "2021;2022;2025", "train_year_count": "3"},
    ]
    result = audit_choice_causality(choices)
    assert result["ok"] is False
    assert result["violations"] == [{"line": "C", "year": 2024, "train_year": 2025}]


def test_trade_year_mismatch_is_detected_from_effective_entry():
    rows = [
        {"line": "D", "trade_id": "D00001", "year": "2020", "entry": "2019-12-31"},
        {"line": "D", "trade_id": "D00002", "year": "2020", "entry": "2020-01-02"},
    ]
    assert trade_year_mismatches(rows) == [
        {"line": "D", "trade_id": "D00001", "stored_year": 2020, "entry_year": 2019}
    ]


def test_release_ok_includes_causality_year_and_overlap_guards():
    line_audits = {
        "C": {
            "missing_prices": 0,
            "bad_dates": 0,
            "duplicate_trade_keys": 0,
            "year_entry_mismatch": 0,
            "overlap_count": 1,
            "all_year_choices_are_causal": True,
        },
        "S": {
            "missing_prices": 0,
            "bad_dates": 0,
            "duplicate_trade_keys": 0,
            "year_entry_mismatch": 0,
            "overlap_count": 0,
            "all_year_choices_are_causal": False,
        },
        "D": {
            "missing_prices": 0,
            "bad_dates": 0,
            "duplicate_trade_keys": 0,
            "year_entry_mismatch": 0,
            "overlap_count": None,
            "all_year_choices_are_causal": True,
        },
        "R": {
            "missing_prices": 0,
            "bad_dates": 0,
            "duplicate_trade_keys": 0,
            "year_entry_mismatch": 0,
            "overlap_count": 0,
            "all_year_choices_are_causal": True,
        },
    }
    assert release_is_ok(line_audits) is False
