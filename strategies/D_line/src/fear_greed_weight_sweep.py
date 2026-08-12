from __future__ import annotations

import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import fear_greed_oos as fg


OUT = fg.OUT / "weight_sweep"
TRAIN_YEARS = (2021, 2022, 2023)
VALID_YEAR = 2025
SHADOW_YEAR = 2026
WEIGHTS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8)
ENTRY_THRESHOLDS = tuple(range(30, 71, 5))
GREED_THRESHOLDS = (65.0, 70.0, 75.0)
MAX_HOLDS = (60, 90)


def composite(core: pd.Series, qvix: pd.Series, weight: float) -> pd.Series:
    return np.where(
        qvix.notna(),
        (1 - weight) * core + weight * qvix,
        core,
    )


def score(frame: pd.DataFrame) -> dict:
    metric = fg.trade_metrics(frame)
    closed = frame[frame["status"].eq("closed") & frame["return_rate"].notna()]
    annual = closed.groupby("test_year")["return_rate"].agg(
        n="size", win=lambda values: float((values > 0).mean())
    )
    annual = annual[annual["n"] >= 10]
    metric["meaningful_worst_year"] = (
        float(annual["win"].min()) if len(annual) else np.nan
    )
    return metric


def generate() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple[float, int, float], pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    raw, pool = fg.load_clean_raw()
    sentiment = fg.build_sentiment(raw)
    symbol_fear = fg.build_symbol_fear(raw)
    trades = pd.read_csv(
        fg.BASE_TRADES, dtype={"symbol": str}, encoding="utf-8-sig"
    )
    trades = fg.rebuild_trade_prices(trades, raw)
    trades = fg.attach_sentiment(trades, sentiment, symbol_fear)

    rows: list[dict] = []
    adjusted_map: dict[tuple[float, int, float], pd.DataFrame] = {}
    for weight in WEIGHTS:
        local_sentiment = sentiment.copy()
        local_sentiment["fear_enhanced"] = composite(
            local_sentiment["fear_core"],
            local_sentiment["fear_qvix"],
            weight,
        )
        local_sentiment["greed_enhanced"] = (
            100 - local_sentiment["fear_enhanced"]
        )
        local_trades = trades.copy()
        local_trades["fear_weighted"] = composite(
            local_trades["fear_core"], local_trades["fear_qvix"], weight
        )
        local_trades["buy_quality"] = (
            0.5 * local_trades["fear_weighted"]
            + 0.5 * local_trades["symbol_fear"]
        )
        for hold in MAX_HOLDS:
            for greed in GREED_THRESHOLDS:
                rule = fg.ExitRule(
                    f"w{weight:.1f}_h{hold}_g{int(greed)}", hold, greed
                )
                adjusted = fg.apply_exit_rule(
                    local_trades, rule, raw, local_sentiment
                )
                adjusted_map[(weight, hold, greed)] = adjusted
                for threshold in ENTRY_THRESHOLDS:
                    selected = adjusted[
                        adjusted["fear_weighted"] >= threshold
                    ].copy()
                    train = selected[
                        selected["test_year"].isin(TRAIN_YEARS)
                    ]
                    rows.append(
                        {
                            "qvix_weight": weight,
                            "entry_threshold": threshold,
                            "max_hold": hold,
                            "greed_threshold": greed,
                            **score(train),
                        }
                    )
    table = pd.DataFrame(rows)
    eligible = (
        (table["closed"] >= 60)
        & (table["covered_years"] >= 2)
        & (table["greed_threshold"] == 70)
    )
    table["selection_score"] = np.where(
        eligible,
        0.50 * table["win_rate"].fillna(0)
        + 0.25 * np.clip(table["avg_return"].fillna(0), -0.10, 0.20)
        + 0.15 * table["meaningful_worst_year"].fillna(0)
        + 0.10 * np.minimum(table["closed"] / 150, 1.0)
        - 0.08 * table["year_win_rate_std"].fillna(1),
        -np.inf,
    )
    return (
        table.sort_values("selection_score", ascending=False),
        sentiment,
        adjusted_map,
        raw,
        pool,
        trades,
    )


def stable(table: pd.DataFrame) -> pd.DataFrame:
    eligible = table[np.isfinite(table["selection_score"])].copy()
    rows: list[pd.Series] = []
    for _, row in eligible.iterrows():
        threshold_neighbors = table[
            table["qvix_weight"].eq(row["qvix_weight"])
            & table["max_hold"].eq(row["max_hold"])
            & table["greed_threshold"].eq(70)
            & table["entry_threshold"].isin(
                [
                    row["entry_threshold"] - 5,
                    row["entry_threshold"],
                    row["entry_threshold"] + 5,
                ]
            )
        ]
        greed_neighbors = table[
            table["qvix_weight"].eq(row["qvix_weight"])
            & table["max_hold"].eq(row["max_hold"])
            & table["entry_threshold"].eq(row["entry_threshold"])
            & table["greed_threshold"].isin(GREED_THRESHOLDS)
        ]
        if len(threshold_neighbors) < 3 or len(greed_neighbors) < 3:
            continue
        if (
            threshold_neighbors["win_rate"].min() >= row["win_rate"] - 0.08
            and threshold_neighbors["avg_return"].min()
            >= row["avg_return"] - 0.04
            and greed_neighbors["avg_return"].min()
            >= row["avg_return"] - 0.05
        ):
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["selection_score", "win_rate", "avg_return"],
        ascending=[False, False, False],
    )


def evaluate_years(
    selected: pd.Series,
    adjusted_map: dict[tuple[float, int, float], pd.DataFrame],
) -> pd.DataFrame:
    key = (
        float(selected["qvix_weight"]),
        int(selected["max_hold"]),
        float(selected["greed_threshold"]),
    )
    frame = adjusted_map[key]
    frame = frame[
        frame["fear_weighted"] >= float(selected["entry_threshold"])
    ].copy()
    rows = []
    for label, years in (
        ("train", TRAIN_YEARS),
        ("validation", (VALID_YEAR,)),
        ("shadow", (SHADOW_YEAR,)),
        ("validation_plus_shadow", (VALID_YEAR, SHADOW_YEAR)),
    ):
        rows.append(
            {
                "stage": label,
                **score(frame[frame["test_year"].isin(years)]),
            }
        )
    return pd.DataFrame(rows), frame


def portfolios(
    selected_trades: pd.DataFrame,
    baseline_trades: pd.DataFrame,
    raw: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for years_label, years in (
        ("all", None),
        ("validation", (VALID_YEAR,)),
        ("shadow", (SHADOW_YEAR,)),
        ("validation_plus_shadow", (VALID_YEAR, SHADOW_YEAR)),
    ):
        for source_name, source in (
            ("fear_greed", selected_trades),
            ("baseline", baseline_trades),
        ):
            local = (
                source
                if years is None
                else source[source["test_year"].isin(years)].copy()
            )
            for slots in (1, 3):
                for ranking in ("liquidity", "fear"):
                    _, summary = fg.portfolio_backtest(
                        local, raw, pool, slots, ranking
                    )
                    rows.append(
                        {
                            "period": years_label,
                            "source": source_name,
                            **summary,
                        }
                    )
    return pd.DataFrame(rows)


def neighbor_robustness(
    adjusted_map: dict[tuple[float, int, float], pd.DataFrame],
    raw: pd.DataFrame,
    pool: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for weight in (0.0, 0.2, 0.3, 0.4):
        for threshold in (40, 45, 50):
            for hold in MAX_HOLDS:
                for greed in GREED_THRESHOLDS:
                    frame = adjusted_map[(weight, hold, greed)]
                    frame = frame[frame["fear_weighted"] >= threshold].copy()
                    for period, years in (
                        ("all", None),
                        ("validation_plus_shadow", (VALID_YEAR, SHADOW_YEAR)),
                    ):
                        local = (
                            frame
                            if years is None
                            else frame[frame["test_year"].isin(years)].copy()
                        )
                        _, summary = fg.portfolio_backtest(
                            local, raw, pool, 1, "fear"
                        )
                        rows.append(
                            {
                                "period": period,
                                "qvix_weight": weight,
                                "entry_threshold": threshold,
                                "max_hold": hold,
                                "greed_threshold": greed,
                                **summary,
                            }
                        )
    return pd.DataFrame(rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    table, sentiment, adjusted_map, raw, pool, _ = generate()
    stable_table = stable(table)
    winner = stable_table.iloc[0]
    yearly, selected_trades = evaluate_years(winner, adjusted_map)
    key = (
        float(winner["qvix_weight"]),
        int(winner["max_hold"]),
        float(winner["greed_threshold"]),
    )
    baseline = adjusted_map[key].copy()
    portfolio = portfolios(selected_trades, baseline, raw, pool)
    neighbors = neighbor_robustness(adjusted_map, raw, pool)

    table.to_csv(OUT / "train_grid.csv", index=False, encoding="utf-8-sig")
    stable_table.to_csv(
        OUT / "stable_grid.csv", index=False, encoding="utf-8-sig"
    )
    yearly.to_csv(OUT / "year_stage.csv", index=False, encoding="utf-8-sig")
    portfolio.to_csv(
        OUT / "portfolio_compare.csv", index=False, encoding="utf-8-sig"
    )
    neighbors.to_csv(
        OUT / "neighbor_portfolios.csv", index=False, encoding="utf-8-sig"
    )
    winner_log, winner_portfolio = fg.portfolio_backtest(
        selected_trades, raw, pool, 1, "fear"
    )
    winner_log.to_csv(
        OUT / "winner_trade_log.csv", index=False, encoding="utf-8-sig"
    )
    holdout_source = selected_trades[
        selected_trades["test_year"].isin((VALID_YEAR, SHADOW_YEAR))
    ].copy()
    holdout_log, holdout_portfolio = fg.portfolio_backtest(
        holdout_source, raw, pool, 1, "fear"
    )
    holdout_log.to_csv(
        OUT / "winner_holdout_trade_log.csv",
        index=False,
        encoding="utf-8-sig",
    )

    latest = sentiment.dropna(subset=["fear_core"]).iloc[-1]
    current_score = float(
        composite(
            pd.Series([latest["fear_core"]]),
            pd.Series([latest["fear_qvix"]]),
            float(winner["qvix_weight"]),
        )[0]
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "winner": {
            "qvix_weight": float(winner["qvix_weight"]),
            "entry_threshold": int(winner["entry_threshold"]),
            "greed_exit": float(winner["greed_threshold"]),
            "max_hold": int(winner["max_hold"]),
        },
        "stage_metrics": yearly.to_dict(orient="records"),
        "latest": {
            "date": str(pd.Timestamp(latest["date"]).date()),
            "fear_score": current_score,
            "buy_filter_pass": current_score
            >= float(winner["entry_threshold"]),
            "greed_exit_pass": 100 - current_score
            >= float(winner["greed_threshold"]),
        },
        "winner_portfolio": winner_portfolio,
        "winner_holdout_portfolio": holdout_portfolio,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
