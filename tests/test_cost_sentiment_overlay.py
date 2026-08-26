from __future__ import annotations

import numpy as np
import pandas as pd

from research.cost_sentiment_overlay.experiment import (
    apply_policy,
    build_chip_features,
    choose_policy_for_year,
)


def _price_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=90)
    close = np.linspace(1.0, 1.3, len(dates))
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "510880",
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(100.0, 220.0, len(dates)),
            "turnover_rate": np.nan,
        }
    )


def test_chip_features_are_causal_when_future_prices_change() -> None:
    original = _price_frame()
    altered = original.copy()
    altered.loc[altered.index[-10:], "close"] *= 3.0
    altered.loc[altered.index[-10:], "high"] = altered.loc[altered.index[-10:], "close"] * 1.01
    altered.loc[altered.index[-10:], "low"] = altered.loc[altered.index[-10:], "close"] * 0.99

    left = build_chip_features(original)
    right = build_chip_features(altered)

    cutoff = len(original) - 10
    columns = ["chip_profit60", "chip_cost_gap60", "chip_overhead5_60", "chip_support5_60"]
    pd.testing.assert_frame_equal(
        left.loc[: cutoff - 1, columns].reset_index(drop=True),
        right.loc[: cutoff - 1, columns].reset_index(drop=True),
    )


def test_s_policy_caps_extension_and_hotness_without_requiring_cheapness() -> None:
    rows = pd.DataFrame(
        {
            "cost_score": [20.0, 20.0, 80.0],
            "cost_extension": [95.0, 70.0, 70.0],
            "market_greed": [50.0, 95.0, 50.0],
            "symbol_hot": [50.0, 50.0, 50.0],
        }
    )
    keep = apply_policy(rows, "S", {"extension_max": 80.0, "greed_max": 90.0})
    assert keep.tolist() == [False, False, True]


def test_annual_selector_does_not_use_test_year_returns() -> None:
    trades = pd.DataFrame(
        {
            "year": [2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024],
            "ret": [0.10, -0.02, 0.08, -0.01, 0.12, -0.03, -0.90, -0.90],
            "cost_score": [80, 20, 80, 20, 80, 20, 80, 20],
            "cost_extension": [20, 90, 20, 90, 20, 90, 20, 90],
            "market_greed": [40, 40, 40, 40, 40, 40, 40, 40],
            "symbol_hot": [40, 40, 40, 40, 40, 40, 40, 40],
        }
    )
    policies = [{"cost_min": 60.0}, {"cost_min": 0.0}]
    first = choose_policy_for_year(trades, "C", 2024, 3, policies)

    changed = trades.copy()
    changed.loc[changed["year"].eq(2024), "ret"] = 9.0
    second = choose_policy_for_year(changed, "C", 2024, 3, policies)

    assert first == second


def test_selector_rejects_sparse_overlay_even_if_sparse_subset_wins() -> None:
    trades = pd.DataFrame(
        {
            "year": [2021] * 5 + [2022] * 5 + [2023] * 5,
            "ret": [0.02] * 14 + [2.0],
            "cost_score": [10.0] * 14 + [100.0],
            "cost_extension": [50.0] * 15,
            "market_greed": [50.0] * 15,
            "symbol_hot": [50.0] * 15,
        }
    )
    policies = [{"cost_min": 90.0}, {"cost_min": 0.0}]
    chosen = choose_policy_for_year(trades, "C", 2024, 3, policies, min_retention=0.70)
    assert chosen == {"cost_min": 0.0}
