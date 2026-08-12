from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from control_hierarchy_test import (
    DEV_YEARS,
    FORMAL_TRADES_PATH,
    HOLDOUT_YEARS,
    ROOT,
    SELECTED_PATH,
    approximate_net,
    generate_trades,
    metrics,
    parse_param,
    rolling_mode,
    score,
)
from shadow_model import ShadowResearchEngine


OUT = ROOT / "fit" / "entry_delay_test"
DELAYS = (0, 1, 2, 3, 4, 5)


def shift_entries(
    engine: ShadowResearchEngine,
    trades: pd.DataFrame,
    delay: int,
) -> pd.DataFrame:
    if trades.empty or delay == 0:
        result = trades.copy()
        if not result.empty:
            result["entry_delay_days"] = delay
            result["selection_year"] = pd.to_datetime(
                result["entry_date"]
            ).dt.year
        return result
    rows: list[dict[str, object]] = []
    for _, trade in trades.iterrows():
        symbol = str(trade["symbol"]).zfill(6)
        frame = engine.panel[symbol]
        entry_pos = engine.pos_after(frame, pd.Timestamp(trade["entry_date"]))
        exit_pos = engine.pos_after(frame, pd.Timestamp(trade["exit_date"]))
        if entry_pos is None or exit_pos is None:
            continue
        new_entry_pos = entry_pos + delay
        if new_entry_pos >= exit_pos or new_entry_pos >= len(frame):
            continue
        new_entry = frame.iloc[new_entry_pos]
        exit_row = frame.iloc[exit_pos]
        item = trade.to_dict()
        item.update(
            {
                "selection_year": pd.Timestamp(trade["entry_date"]).year,
                "entry_date": pd.Timestamp(new_entry["date"]),
                "entry_close": float(new_entry["close"]),
                "exit_close": float(exit_row["close"]),
                "ret": float(exit_row["close"] / new_entry["close"] - 1.0),
                "entry_delay_days": delay,
            }
        )
        rows.append(item)
    return pd.DataFrame(rows)


def build_account(
    engine: ShadowResearchEngine,
    annual: pd.DataFrame,
    shifted: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    pieces = [engine.main.copy()]
    for _, selected in annual.iterrows():
        param_id = str(selected["param_id"])
        if param_id == "CASH":
            continue
        year = int(selected["year"])
        frame = shifted[param_id]
        pieces.append(frame[frame["selection_year"].eq(year)].copy())
    return engine.account_from_trades(
        pd.concat(pieces, ignore_index=True, sort=False)
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    engine = ShadowResearchEngine()
    annual = pd.read_csv(
        SELECTED_PATH, dtype={"param_id": str}, encoding="utf-8-sig"
    )
    params = {
        param_id: parse_param(param_id)
        for param_id in annual["param_id"].dropna().unique()
        if param_id != "CASH"
    }
    base = {
        param_id: generate_trades(engine, param, "flat")
        for param_id, param in params.items()
    }
    accounts: dict[str, pd.DataFrame] = {}
    fixed_rows: list[dict[str, object]] = []
    shifted_map: dict[int, dict[str, pd.DataFrame]] = {}
    for delay in DELAYS:
        key = "flat" if delay == 0 else f"delay_{delay}"
        shifted = {
            param_id: shift_entries(engine, trades, delay)
            for param_id, trades in base.items()
        }
        shifted_map[delay] = shifted
        account_frame = build_account(engine, annual, shifted)
        accounts[key] = account_frame
        dev = metrics(account_frame, DEV_YEARS)
        holdout = metrics(account_frame, HOLDOUT_YEARS)
        fixed_rows.append(
            {
                "entry_mode": key,
                "extra_delay_days": delay,
                "dev_score": score(dev),
                **{f"dev_{name}": value for name, value in dev.items()},
                **{f"holdout_{name}": value for name, value in holdout.items()},
                "approx_net_final_10000": approximate_net(account_frame),
            }
        )
        print(f"[S entry delay] {delay}")
    fixed = pd.DataFrame(fixed_rows).sort_values(
        ["dev_score", "dev_final_1000"], ascending=False
    )

    methods: list[dict[str, object]] = []
    choices_map: dict[str, pd.DataFrame] = {}
    logs_map: dict[str, pd.DataFrame] = {}
    for window in (3, 5, 7, None):
        method = f"w{window if window is not None else 'all'}"
        choices, log = rolling_mode(accounts, window, engine)
        choices_map[method] = choices
        logs_map[method] = log
        dev = metrics(log, DEV_YEARS)
        holdout = metrics(log, HOLDOUT_YEARS)
        methods.append(
            {
                "method": method,
                "dev_score": score(dev),
                **{f"dev_{name}": value for name, value in dev.items()},
                **{f"holdout_{name}": value for name, value in holdout.items()},
                "approx_net_final_10000": approximate_net(log),
            }
        )
    methods_frame = pd.DataFrame(methods).sort_values(
        ["dev_score", "dev_final_1000"], ascending=False
    )
    winner = str(methods_frame.iloc[0]["method"])

    formal = pd.read_csv(
        FORMAL_TRADES_PATH, dtype={"symbol": str}, encoding="utf-8-sig"
    )
    formal["entry_date"] = pd.to_datetime(formal["entry_date"])
    baseline_holdout = metrics(formal, HOLDOUT_YEARS)
    baseline_reproduction = metrics(
        accounts["flat"], DEV_YEARS | HOLDOUT_YEARS
    )
    winner_holdout = metrics(logs_map[winner], HOLDOUT_YEARS)
    pass_count = int(
        (
            (methods_frame["holdout_final_1000"] >= baseline_holdout["final_1000"])
            & (
                methods_frame["holdout_win_rate"]
                >= float(baseline_holdout["win_rate"]) - 0.02
            )
            & (
                methods_frame["holdout_max_drawdown"]
                >= float(baseline_holdout["max_drawdown"]) - 0.02
            )
        ).sum()
    )
    promoted = bool(
        winner_holdout["final_1000"] > baseline_holdout["final_1000"]
        and winner_holdout["win_rate"] >= baseline_holdout["win_rate"]
        and winner_holdout["max_drawdown"] >= baseline_holdout["max_drawdown"]
        and pass_count >= 2
    )

    fixed.to_csv(OUT / "variants.csv", index=False, encoding="utf-8-sig")
    methods_frame.to_csv(
        OUT / "rolling_methods.csv", index=False, encoding="utf-8-sig"
    )
    choices_map[winner].to_csv(
        OUT / "winner_choices.csv", index=False, encoding="utf-8-sig"
    )
    logs_map[winner].to_csv(
        OUT / "winner_trades.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "control_variable": "S1_entry_execution_delay_only",
        "signal_exit_and_holding_window_unchanged": True,
        "formal_baseline_holdout": baseline_holdout,
        "baseline_reproduction_all": baseline_reproduction,
        "development_selected_method": winner,
        "winner_holdout": winner_holdout,
        "winner_approx_net_final_10000": approximate_net(logs_map[winner]),
        "neighbor_methods_passing_holdout_floor": pass_count,
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
