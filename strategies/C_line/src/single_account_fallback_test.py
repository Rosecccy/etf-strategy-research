from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
S_SRC = WORKSPACE / "S" / "src"
if str(S_SRC) not in sys.path:
    sys.path.insert(0, str(S_SRC))

from shadow_model import ShadowResearchEngine  # noqa: E402
from strengthen_selector import (  # noqa: E402
    DEV_YEARS,
    HOLDOUT_YEARS,
    generate_candidates,
    s_params,
    segment_metrics,
    test_methods,
)


OUT = ROOT / "fit" / "single_account_fallback_test"
BASELINE_PATH = ROOT / "fit" / "fallback" / "final_trades.csv"


def load_c_account() -> pd.DataFrame:
    frame = pd.read_csv(
        BASELINE_PATH,
        dtype={"symbol": str},
        encoding="utf-8-sig",
    )
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    frame["exit_date"] = pd.to_datetime(frame["exit_date"], errors="coerce")
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    frame["source_type"] = frame.get("source_type", frame["source"])
    frame["priority"] = pd.to_numeric(
        frame.get("priority", 1000), errors="coerce"
    ).fillna(1000)
    return frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    engine = ShadowResearchEngine()
    baseline = load_c_account()
    # The entire real C account is the only occupancy clock.
    engine.main = baseline.copy()
    engine.idle_windows = engine._build_idle_windows()

    trades_cache = OUT / "candidate_trades.parquet"
    yearly_cache = OUT / "candidate_yearly.parquet"
    if trades_cache.exists() and yearly_cache.exists():
        candidate_trades = pd.read_parquet(trades_cache)
        candidate_yearly = pd.read_parquet(yearly_cache)
        for column in ("decision_date", "entry_date", "exit_date"):
            if column in candidate_trades:
                candidate_trades[column] = pd.to_datetime(
                    candidate_trades[column], errors="coerce"
                )
        print("[cache] loaded candidate trades", flush=True)
    else:
        params = s_params(engine)
        candidate_trades, candidate_yearly = generate_candidates(engine, params)
        candidate_trades.to_parquet(trades_cache, index=False)
        candidate_yearly.to_parquet(yearly_cache, index=False)
        pd.DataFrame(
            [
                {
                    **param.__dict__,
                    "param_id": param.param_id,
                }
                for param in params
            ]
        ).to_csv(
            OUT / "candidate_params.csv",
            index=False,
            encoding="utf-8-sig",
        )

    methods, choices, accounts = test_methods(
        engine,
        candidate_yearly,
        candidate_trades,
    )
    winner = methods.iloc[0]
    winner_id = str(winner["method_id"])
    winner_choices = choices[winner_id]
    winner_trades = accounts[winner_id]
    baseline_dev = segment_metrics(baseline, DEV_YEARS)
    baseline_holdout = segment_metrics(baseline, HOLDOUT_YEARS)

    holdout_floor = (
        (methods["holdout_final_cash"] >= baseline_holdout["final_cash"] * 1.05)
        & (
            methods["holdout_win_rate"]
            >= float(baseline_holdout["win_rate"]) - 0.02
        )
        & (
            methods["holdout_max_drawdown"]
            >= float(baseline_holdout["max_drawdown"]) - 0.02
        )
        & (methods["holdout_trades"] >= 10)
    )
    neighbors = int(holdout_floor.sum())
    promoted = bool(
        float(winner["holdout_final_cash"])
        >= float(baseline_holdout["final_cash"]) * 1.05
        and float(winner["holdout_win_rate"])
        >= float(baseline_holdout["win_rate"]) - 0.02
        and float(winner["holdout_max_drawdown"])
        >= float(baseline_holdout["max_drawdown"]) - 0.02
        and int(winner["holdout_trades"]) >= 10
        and neighbors >= 2
    )

    methods.to_csv(
        OUT / "method_grid.csv", index=False, encoding="utf-8-sig"
    )
    winner_choices.to_csv(
        OUT / "winner_choices.csv", index=False, encoding="utf-8-sig"
    )
    winner_trades.to_csv(
        OUT / "winner_trades.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": "single C account plus cross-asset fallback",
        "account_clock": "full C formal account, strict causal idle",
        "candidate_params": int(candidate_trades["param_id"].nunique()),
        "method_variants": int(len(methods)),
        "baseline_development": baseline_dev,
        "baseline_holdout": baseline_holdout,
        "development_selected_method": winner_id,
        "winner": winner.to_dict(),
        "holdout_neighbors_passing": neighbors,
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
