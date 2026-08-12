from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from control_exit_test import (
    DEV_YEARS,
    ETF_DIR,
    HOLDOUT_YEARS,
    ROOT,
    TRADES_PATH,
    load_prices,
    metrics,
    net_account,
    rolling_choices,
    selection_score,
)
from control_regime_test import RegimeRule, build_log, load_market_state


OUT = ROOT / "fit" / "regime_duration_test"
DURATIONS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(
        TRADES_PATH, dtype={"symbol": str}, encoding="utf-8-sig"
    )
    source["symbol"] = source["symbol"].astype(str).str.zfill(6)
    source["entry_date"] = pd.to_datetime(source["entry_date"])
    source["exit_date"] = pd.to_datetime(source["exit_date"])
    source["ret"] = pd.to_numeric(source["ret"], errors="coerce")
    price_map = {
        symbol: load_prices(symbol)
        for symbol in sorted(source["symbol"].unique())
    }
    state = load_market_state()

    logs = {
        "baseline": build_log(
            source,
            price_map,
            state,
            RegimeRule("baseline", "baseline"),
            0,
        )
    }
    fixed_rows: list[dict[str, object]] = []
    baseline_dev = metrics(logs["baseline"], DEV_YEARS)
    baseline_holdout = metrics(logs["baseline"], HOLDOUT_YEARS)
    fixed_rows.append(
        {
            "rule": "baseline",
            "extend_days": 0,
            "dev_score": selection_score(baseline_dev),
            **{f"dev_{key}": value for key, value in baseline_dev.items()},
            **{
                f"holdout_{key}": value
                for key, value in baseline_holdout.items()
            },
        }
    )
    for days in DURATIONS:
        key = f"market_ret20_02_extend_{days}"
        log = build_log(
            source,
            price_map,
            state,
            RegimeRule(key, "ret20", 0.02),
            days,
        )
        logs[key] = log
        dev = metrics(log, DEV_YEARS)
        holdout = metrics(log, HOLDOUT_YEARS)
        fixed_rows.append(
            {
                "rule": key,
                "extend_days": days,
                "dev_score": selection_score(dev),
                **{f"dev_{name}": value for name, value in dev.items()},
                **{f"holdout_{name}": value for name, value in holdout.items()},
            }
        )
        print(f"[C duration] {key}")
    fixed = pd.DataFrame(fixed_rows).sort_values(
        ["dev_score", "dev_final_1000"], ascending=False
    )

    methods: list[dict[str, object]] = []
    choices_map: dict[str, pd.DataFrame] = {}
    logs_map: dict[str, pd.DataFrame] = {}
    for window in (3, 5, 7, None):
        method = f"w{window if window is not None else 'all'}"
        choices, log = rolling_choices(logs, window)
        choices_map[method] = choices
        logs_map[method] = log
        dev = metrics(log, DEV_YEARS)
        holdout = metrics(log, HOLDOUT_YEARS)
        methods.append(
            {
                "method": method,
                "dev_score": selection_score(dev),
                **{f"dev_{name}": value for name, value in dev.items()},
                **{f"holdout_{name}": value for name, value in holdout.items()},
            }
        )
    methods_frame = pd.DataFrame(methods).sort_values(
        ["dev_score", "dev_final_1000"], ascending=False
    )
    winner = str(methods_frame.iloc[0]["method"])
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
    _, baseline_net = net_account(logs["baseline"])
    winner_net_log, winner_net = net_account(logs_map[winner])

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
    winner_net_log.to_csv(
        OUT / "winner_net_trades.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "control_variable": "strong_market_exit_extension_days_only",
        "fixed_regime": "equal_weight ETF median 20-day return >= 2%",
        "baseline_holdout": baseline_holdout,
        "baseline_net": baseline_net,
        "development_selected_method": winner,
        "winner_holdout": winner_holdout,
        "winner_net": winner_net,
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
