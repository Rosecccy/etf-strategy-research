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


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "control_test"
DEV_YEARS = (2021, 2022, 2023)
HOLDOUT_YEARS = (2025, 2026)


def sizing_fraction(mode: str, fear_score: float) -> float:
    score = float(np.clip(fear_score, 0.0, 100.0))
    if mode == "full":
        return 1.0
    if mode == "step_50":
        return 0.50 if score < 55 else (0.75 if score < 65 else 1.0)
    if mode == "step_60":
        return 0.60 if score < 55 else (0.80 if score < 65 else 1.0)
    if mode == "step_75":
        return 0.75 if score < 55 else (0.90 if score < 65 else 1.0)
    if mode == "linear_50":
        return float(np.clip(0.50 + (score - 45.0) / 50.0, 0.50, 1.0))
    if mode == "linear_60":
        return float(np.clip(0.60 + (score - 45.0) / 62.5, 0.60, 1.0))
    if mode == "linear_75":
        return float(np.clip(0.75 + (score - 45.0) / 100.0, 0.75, 1.0))
    raise ValueError(f"Unknown sizing mode: {mode}")


def account(
    candidates: pd.DataFrame,
    raw: pd.DataFrame,
    pool: pd.DataFrame,
    mode: str,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    local = candidates.copy()
    local["entry_date"] = pd.to_datetime(local["entry_date"])
    local["exit_date"] = pd.to_datetime(local["exit_date"], errors="coerce")
    liquidity = (
        pool.set_index("symbol")["amount20"]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_dict()
    )
    local["liquidity"] = local["symbol"].map(liquidity).fillna(0)
    local = local.sort_values(
        ["entry_date", "buy_quality", "liquidity", "symbol"],
        ascending=[True, False, False, True],
    )
    by_entry = {
        date: group for date, group in local.groupby("entry_date", sort=False)
    }
    dates = sorted(
        set(local["entry_date"].dropna()) | set(local["exit_date"].dropna())
    )
    price_map = raw.set_index(["symbol", "date"])["close"]
    latest_price = (
        raw.sort_values("date")
        .groupby("symbol", sort=False)
        .tail(1)
        .set_index("symbol")["close"]
    )
    latest_date = pd.Timestamp(raw["date"].max())
    cash = fg.INITIAL_CAPITAL
    position: dict[str, object] | None = None
    records: list[dict[str, object]] = []
    equity_marks = [cash]

    for date in dates:
        if (
            position is not None
            and pd.notna(position["exit_date"])
            and pd.Timestamp(position["exit_date"]) == date
        ):
            price = float(
                price_map.get(
                    (position["symbol"], date), position["exit_close"]
                )
            )
            proceeds = int(position["quantity"]) * price
            fee = fg.commission(proceeds)
            cash += proceeds - fee
            net_return = (
                proceeds - fee - float(position["cost_total"])
            ) / float(position["cost_total"])
            record = dict(position)
            record.update(
                {
                    "actual_exit_date": date.date().isoformat(),
                    "actual_exit_close": price,
                    "sell_fee": fee,
                    "net_return": net_return,
                    "net_pnl": proceeds
                    - fee
                    - float(position["cost_total"]),
                    "status_portfolio": "closed",
                    "equity_after": cash,
                }
            )
            records.append(record)
            equity_marks.append(cash)
            position = None

        if position is not None or date not in by_entry:
            continue
        trade = by_entry[date].iloc[0]
        price = float(trade["entry_close"])
        fear_score = float(trade["fear_weighted"])
        fraction = sizing_fraction(mode, fear_score)
        target = cash * fraction
        quantity = math.floor(
            (target - fg.MIN_COMMISSION) / (price * 100)
        ) * 100
        if quantity < 100:
            continue
        notional = quantity * price
        buy_fee = fg.commission(notional)
        if notional + buy_fee > cash:
            continue
        cash -= notional + buy_fee
        position = {
            "symbol": str(trade["symbol"]),
            "name": trade.get("name", ""),
            "entry_date": date.date().isoformat(),
            "entry_close": price,
            "exit_date": trade["exit_date"],
            "exit_close": trade.get("exit_close", np.nan),
            "quantity": int(quantity),
            "buy_fee": buy_fee,
            "cost_total": notional + buy_fee,
            "fear_score": fear_score,
            "target_fraction": fraction,
            "sizing_mode": mode,
        }

    open_value = 0.0
    if position is not None:
        mark = float(
            latest_price.get(position["symbol"], position["entry_close"])
        )
        fee = fg.commission(int(position["quantity"]) * mark)
        open_value = int(position["quantity"]) * mark - fee
        record = dict(position)
        record.update(
            {
                "actual_exit_date": "",
                "actual_exit_close": mark,
                "sell_fee": fee,
                "net_return": (
                    open_value - float(position["cost_total"])
                )
                / float(position["cost_total"]),
                "net_pnl": open_value - float(position["cost_total"]),
                "status_portfolio": "open_marked",
                "equity_after": cash + open_value,
            }
        )
        records.append(record)
        equity_marks.append(cash + open_value)

    log = pd.DataFrame(records)
    closed = (
        log[log["status_portfolio"].eq("closed")]
        if not log.empty
        else log
    )
    equity = pd.Series(equity_marks, dtype=float)
    drawdown = equity / equity.cummax() - 1.0
    final_value = cash + open_value
    return log, {
        "initial_capital": fg.INITIAL_CAPITAL,
        "final_value": float(final_value),
        "net_profit": float(final_value - fg.INITIAL_CAPITAL),
        "total_return": float(final_value / fg.INITIAL_CAPITAL - 1.0),
        "accepted_trades": int(len(log)),
        "closed_trades": int(len(closed)),
        "win_rate": float((closed["net_return"] > 0).mean())
        if len(closed)
        else np.nan,
        "avg_net_return": float(closed["net_return"].mean())
        if len(closed)
        else np.nan,
        "max_drawdown": float(drawdown.min()),
        "open_positions": int(position is not None),
        "cash": float(cash),
        "open_mark_value": float(open_value),
        "as_of": latest_date.date().isoformat(),
    }


def score(summary: dict[str, float | int]) -> float:
    return (
        math.log(
            max(float(summary["final_value"]), 1.0) / fg.INITIAL_CAPITAL
        )
        + 0.25
        * float(np.nan_to_num(summary["win_rate"], nan=0.0))
        + 0.70 * float(summary["max_drawdown"])
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    table, _, adjusted_map, raw, pool, _ = sweep.generate()
    stable = sweep.stable(table)
    winner = stable.iloc[0]
    key = (
        float(winner["qvix_weight"]),
        int(winner["max_hold"]),
        float(winner["greed_threshold"]),
    )
    selected = adjusted_map[key].copy()
    selected = selected[
        selected["fear_weighted"]
        >= float(winner["entry_threshold"])
    ].copy()
    selected["overlay"] = "fear_greed"

    modes = (
        "full",
        "step_50",
        "step_60",
        "step_75",
        "linear_50",
        "linear_60",
        "linear_75",
    )
    rows: list[dict[str, object]] = []
    logs: dict[tuple[str, str], pd.DataFrame] = {}
    summaries: dict[tuple[str, str], dict[str, float | int]] = {}
    for mode in modes:
        for stage, years in (
            ("development", DEV_YEARS),
            ("holdout", HOLDOUT_YEARS),
            ("all", None),
        ):
            source = (
                selected
                if years is None
                else selected[selected["test_year"].isin(years)].copy()
            )
            log, summary = account(source, raw, pool, mode)
            logs[(mode, stage)] = log
            summaries[(mode, stage)] = summary
            rows.append(
                {
                    "sizing_mode": mode,
                    "stage": stage,
                    "selection_score": score(summary),
                    **summary,
                }
            )
        print(f"[D sizing] {mode}")
    results = pd.DataFrame(rows)
    development = results[results["stage"].eq("development")].sort_values(
        ["selection_score", "final_value"], ascending=False
    )
    chosen = str(development.iloc[0]["sizing_mode"])
    baseline_holdout = summaries[("full", "holdout")]
    winner_holdout = summaries[(chosen, "holdout")]
    pass_count = int(
        sum(
            summaries[(mode, "holdout")]["final_value"]
            >= baseline_holdout["final_value"]
            and summaries[(mode, "holdout")]["max_drawdown"]
            >= baseline_holdout["max_drawdown"] - 0.01
            for mode in modes
        )
    )
    promoted = bool(
        winner_holdout["final_value"] > baseline_holdout["final_value"]
        and winner_holdout["win_rate"] >= baseline_holdout["win_rate"]
        and winner_holdout["max_drawdown"]
        >= baseline_holdout["max_drawdown"]
        and pass_count >= 2
    )

    results.to_csv(
        OUT / "sizing_variants.csv", index=False, encoding="utf-8-sig"
    )
    logs[(chosen, "all")].to_csv(
        OUT / "winner_trades.csv", index=False, encoding="utf-8-sig"
    )
    logs[(chosen, "holdout")].to_csv(
        OUT / "winner_holdout_trades.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "control_variable": "fear_score_position_sizing_only",
        "technical_signal_and_exit_unchanged": True,
        "fear_greed_params": {
            "qvix_weight": float(winner["qvix_weight"]),
            "entry_threshold": float(winner["entry_threshold"]),
            "max_hold": int(winner["max_hold"]),
            "greed_threshold": float(winner["greed_threshold"]),
        },
        "development_selected_mode": chosen,
        "baseline_holdout": baseline_holdout,
        "winner_holdout": winner_holdout,
        "holdout_modes_passing_floor": pass_count,
        "promoted": promoted,
        "formal_config_changed": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
