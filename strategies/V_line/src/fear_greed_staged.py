from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import fear_greed_oos as fg


OUT = fg.OUT / "staged"
TRAIN_YEARS = (2021, 2022, 2023)
VALID_YEARS = (2025, 2026)


def candidate_frame(
    trades: pd.DataFrame,
    adjusted: dict[str, pd.DataFrame],
    overlay: fg.Overlay,
    exit_rule: fg.ExitRule,
    years: tuple[int, ...],
) -> pd.DataFrame:
    frame = adjusted[exit_rule.key]
    frame = frame[frame["test_year"].isin(years)].copy()
    return frame[fg.overlay_mask(frame, overlay).fillna(False)].copy()


def meaningful_worst_year(frame: pd.DataFrame) -> float:
    closed = frame[frame["status"].eq("closed") & frame["return_rate"].notna()]
    annual = closed.groupby("test_year")["return_rate"].agg(
        trades="size", win_rate=lambda values: float((values > 0).mean())
    )
    annual = annual[annual["trades"] >= 10]
    return float(annual["win_rate"].min()) if len(annual) else np.nan


def score_grid(
    trades: pd.DataFrame,
    adjusted: dict[str, pd.DataFrame],
    overlays: list[fg.Overlay],
    exits: list[fg.ExitRule],
    years: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict] = []
    for exit_rule in exits:
        for overlay in overlays:
            frame = candidate_frame(trades, adjusted, overlay, exit_rule, years)
            metric = fg.trade_metrics(frame)
            rows.append(
                {
                    "overlay": overlay.key,
                    "family": overlay.family,
                    "threshold": overlay.threshold,
                    "recovery_change": overlay.recovery_change,
                    "exit_key": exit_rule.key,
                    "max_hold": exit_rule.max_hold,
                    "greed_threshold": exit_rule.greed_threshold,
                    "meaningful_worst_year": meaningful_worst_year(frame),
                    **metric,
                }
            )
    table = pd.DataFrame(rows)
    eligible = (
        (table["closed"] >= 60)
        & (table["covered_years"] >= 2)
        & table["family"].isin(["baseline", "core", "enhanced", "qvix", "recovery"])
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
    return table.sort_values(
        ["selection_score", "win_rate", "avg_return", "closed"],
        ascending=[False, False, False, False],
    )


def stable_candidates(table: pd.DataFrame) -> pd.DataFrame:
    eligible = table[np.isfinite(table["selection_score"])].copy()
    stable_rows: list[pd.Series] = []
    for _, row in eligible.iterrows():
        if row["family"] == "baseline":
            stable_rows.append(row)
            continue
        step = 10 if row["family"] == "qvix" else 5
        neighbors = eligible[
            eligible["family"].eq(row["family"])
            & eligible["exit_key"].eq(row["exit_key"])
            & eligible["threshold"].isin(
                [row["threshold"] - step, row["threshold"], row["threshold"] + step]
            )
        ]
        if row["family"] == "recovery":
            neighbors = neighbors[
                neighbors["recovery_change"].eq(row["recovery_change"])
            ]
        if len(neighbors) < 3:
            continue
        if (
            neighbors["win_rate"].min() >= row["win_rate"] - 0.08
            and neighbors["avg_return"].min() >= row["avg_return"] - 0.04
        ):
            stable_rows.append(row)
    if not stable_rows:
        return eligible.head(1)
    return pd.DataFrame(stable_rows).sort_values(
        ["selection_score", "win_rate", "avg_return", "closed"],
        ascending=[False, False, False, False],
    )


def evaluate_selected(
    candidates: pd.DataFrame,
    trades: pd.DataFrame,
    adjusted: dict[str, pd.DataFrame],
    overlays: list[fg.Overlay],
    exits: list[fg.ExitRule],
) -> pd.DataFrame:
    overlay_map = {item.key: item for item in overlays}
    exit_map = {item.key: item for item in exits}
    rows: list[dict] = []
    for rank, (_, selected) in enumerate(candidates.head(20).iterrows(), start=1):
        overlay = overlay_map[str(selected["overlay"])]
        exit_rule = exit_map[str(selected["exit_key"])]
        valid = candidate_frame(
            trades, adjusted, overlay, exit_rule, VALID_YEARS
        )
        rows.append(
            {
                "train_rank": rank,
                "overlay": overlay.key,
                "family": overlay.family,
                "exit_key": exit_rule.key,
                "train_closed": int(selected["closed"]),
                "train_win_rate": float(selected["win_rate"]),
                "train_avg_return": float(selected["avg_return"]),
                **{f"valid_{key}": value for key, value in fg.trade_metrics(valid).items()},
            }
        )
    return pd.DataFrame(rows)


def portfolio_scenarios(
    trades: pd.DataFrame,
    raw: pd.DataFrame,
    pool: pd.DataFrame,
    overlay: fg.Overlay,
    exit_rule: fg.ExitRule,
    adjusted: dict[str, pd.DataFrame],
    years: tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    selected = adjusted[exit_rule.key].copy()
    if years is not None:
        selected = selected[selected["test_year"].isin(years)].copy()
    selected = selected[fg.overlay_mask(selected, overlay).fillna(False)].copy()
    selected["overlay"] = overlay.key
    baseline = adjusted[exit_rule.key].copy()
    if years is not None:
        baseline = baseline[baseline["test_year"].isin(years)].copy()
    baseline["overlay"] = "baseline"
    rows: list[dict] = []
    logs: dict[str, pd.DataFrame] = {}
    for source_name, source in (("fear_greed", selected), ("baseline", baseline)):
        for slots in (1, 3):
            for ranking in ("liquidity", "fear"):
                log, summary = fg.portfolio_backtest(
                    source, raw, pool, slots, ranking
                )
                key = f"{source_name}_{slots}slot_{ranking}"
                logs[key] = log
                rows.append({"scenario": key, **summary})
    return pd.DataFrame(rows).sort_values("final_value", ascending=False), logs


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    raw, pool = fg.load_clean_raw()
    sentiment = fg.build_sentiment(raw)
    symbol_fear = fg.build_symbol_fear(raw)
    trades = pd.read_csv(
        fg.BASE_TRADES, dtype={"symbol": str}, encoding="utf-8-sig"
    )
    trades = fg.rebuild_trade_prices(trades, raw)
    trades = fg.attach_sentiment(trades, sentiment, symbol_fear)
    overlays = fg.overlay_grid()
    exits = fg.exit_grid()
    adjusted = {
        rule.key: fg.apply_exit_rule(trades, rule, raw, sentiment)
        for rule in exits
    }

    training = score_grid(trades, adjusted, overlays, exits, TRAIN_YEARS)
    holdout_all = score_grid(trades, adjusted, overlays, exits, VALID_YEARS)
    stable = stable_candidates(training)
    validation = evaluate_selected(
        stable, trades, adjusted, overlays, exits
    )
    training.to_csv(OUT / "train_grid.csv", index=False, encoding="utf-8-sig")
    holdout_all.to_csv(
        OUT / "holdout_grid.csv", index=False, encoding="utf-8-sig"
    )
    stable.to_csv(OUT / "stable_train.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(OUT / "holdout.csv", index=False, encoding="utf-8-sig")

    winner = stable.iloc[0]
    overlay = next(item for item in overlays if item.key == winner["overlay"])
    exit_rule = next(item for item in exits if item.key == winner["exit_key"])
    portfolios, logs = portfolio_scenarios(
        trades, raw, pool, overlay, exit_rule, adjusted
    )
    portfolios.to_csv(
        OUT / "portfolio_compare.csv", index=False, encoding="utf-8-sig"
    )
    for key, log in logs.items():
        log.to_csv(OUT / f"{key}.csv", index=False, encoding="utf-8-sig")
    holdout_portfolios, holdout_logs = portfolio_scenarios(
        trades, raw, pool, overlay, exit_rule, adjusted, VALID_YEARS
    )
    holdout_portfolios.to_csv(
        OUT / "holdout_portfolio_compare.csv",
        index=False,
        encoding="utf-8-sig",
    )
    for key, log in holdout_logs.items():
        log.to_csv(
            OUT / f"holdout_{key}.csv", index=False, encoding="utf-8-sig"
        )

    live_training = score_grid(
        trades,
        adjusted,
        overlays,
        exits,
        (2021, 2022, 2023, 2025),
    )
    live_stable = stable_candidates(live_training)
    live_winner = live_stable.iloc[0]
    latest = sentiment.dropna(subset=["fear_enhanced"]).iloc[-1]
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_end": str(raw["date"].max().date()),
        "train_years": list(TRAIN_YEARS),
        "holdout_years": list(VALID_YEARS),
        "research_winner": {
            "overlay": str(winner["overlay"]),
            "exit_key": str(winner["exit_key"]),
            "train_closed": int(winner["closed"]),
            "train_win_rate": float(winner["win_rate"]),
            "train_avg_return": float(winner["avg_return"]),
        },
        "holdout": validation.iloc[0].to_dict(),
        "live_2026_choice_using_prior_years": {
            "overlay": str(live_winner["overlay"]),
            "exit_key": str(live_winner["exit_key"]),
        },
        "latest_sentiment": {
            "date": str(pd.Timestamp(latest["date"]).date()),
            "fear_core": float(latest["fear_core"]),
            "fear_qvix": float(latest["fear_qvix"]),
            "fear_enhanced": float(latest["fear_enhanced"]),
            "greed_enhanced": float(latest["greed_enhanced"]),
            "regime": str(latest["regime"]),
        },
        "best_portfolio": portfolios.iloc[0].to_dict(),
        "best_holdout_portfolio": holdout_portfolios.iloc[0].to_dict(),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
