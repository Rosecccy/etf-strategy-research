from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from shadow_model import Param, ShadowResearchEngine


ROOT = Path(__file__).resolve().parents[2]
WINDOWS = ([3], [5], [7], [3, 5], [5, 7])
SCORE_KINDS = ("stable", "balanced", "anti_conc", "robust")
CASH_GATES = (-0.02, 0.0, 0.02, 0.04)
TEST_YEARS = list(range(2016, 2027))
DEV_YEARS = set(range(2016, 2024))
HOLDOUT_YEARS = {2024, 2025, 2026}


def unique_params(params: list[Param]) -> list[Param]:
    return list({param.param_id: param for param in params}.values())


def c_params(engine: ShadowResearchEngine) -> list[Param]:
    engine.scope_map["red_only"] = ("510880",)
    params: list[Param] = []
    for idle in (10, 20, 40):
        for hold in (10, 20, 40):
            for ma in (40, 60, 120):
                for mom in (20, 60):
                    params.append(
                        Param(
                            "red_only",
                            idle,
                            hold,
                            ma,
                            mom,
                            "raw",
                            0.0,
                            0.6,
                            1.0,
                            250,
                            2_000_000,
                        )
                    )
    for idle in (20, 40):
        for hold in (10, 40, 60):
            for ma in (60, 120):
                for mom in (20, 60):
                    for model in ("raw", "stable", "consistent"):
                        params.append(
                            Param(
                                "domestic_core",
                                idle,
                                hold,
                                ma,
                                mom,
                                model,
                                0.0 if model == "raw" else 0.4,
                                0.6,
                                1.0,
                                250,
                                2_000_000,
                            )
                        )
    return unique_params(params)


def s_params(_: ShadowResearchEngine) -> list[Param]:
    params: list[Param] = []
    scopes = (
        "domestic_plus_precious",
        "domestic_plus_us",
        "domestic_plus_precious_us",
        "all_clean",
    )
    for scope in scopes:
        for idle in (20, 40, 60):
            for hold in (10, 40, 60):
                for ma in (60, 120):
                    for mom in (20, 60):
                        for model in ("raw", "stable", "consistent"):
                            params.append(
                                Param(
                                    scope,
                                    idle,
                                    hold,
                                    ma,
                                    mom,
                                    model,
                                    0.0 if model == "raw" else 0.4,
                                    0.6,
                                    1.0,
                                    250,
                                    2_000_000,
                                )
                            )
    return unique_params(params)


def generate_candidates(
    engine: ShadowResearchEngine,
    params: list[Param],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logs: list[pd.DataFrame] = []
    for index, param in enumerate(params, start=1):
        log = engine.generate_shadow_trades(param, causal_idle=True)
        if not log.empty:
            logs.append(log)
        if index == 1 or index % 25 == 0 or index == len(params):
            print(f"[candidate {index}/{len(params)}] {param.param_id}")
    trades = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    yearly = engine.param_yearly(trades)
    return trades, yearly


def selected_account(
    engine: ShadowResearchEngine,
    selected: pd.DataFrame,
    candidate_trades: pd.DataFrame,
) -> pd.DataFrame:
    pieces = [engine.main.copy()]
    trade_year = pd.to_datetime(
        candidate_trades["entry_date"], errors="coerce"
    ).dt.year
    for _, choice in selected.iterrows():
        if str(choice["param_id"]) == "CASH":
            continue
        year = int(choice["test_year"])
        pieces.append(
            candidate_trades[
                candidate_trades["param_id"].eq(choice["param_id"])
                & trade_year.eq(year)
            ].copy()
        )
    return engine.account_from_trades(
        pd.concat(pieces, ignore_index=True, sort=False)
    )


def segment_metrics(account: pd.DataFrame, years: set[int]) -> dict[str, object]:
    local = account.copy()
    local["entry_year"] = pd.to_datetime(
        local["entry_date"], errors="coerce"
    ).dt.year
    local = local[local["entry_year"].isin(years)].copy()
    if local.empty:
        return {
            "trades": 0,
            "final_cash": 1000.0,
            "total_return": 0.0,
            "win_rate": np.nan,
            "avg_return": np.nan,
            "max_drawdown": 0.0,
            "underperform_years": 0,
        }
    returns = pd.to_numeric(local["ret"], errors="coerce").dropna()
    equity = 1000.0 * (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    annual = (
        local.assign(ret_num=pd.to_numeric(local["ret"], errors="coerce"))
        .groupby("entry_year")["ret_num"]
        .apply(lambda values: float(np.prod(1.0 + values.dropna()) - 1.0))
    )
    return {
        "trades": int(len(returns)),
        "final_cash": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / 1000.0 - 1.0),
        "win_rate": float((returns > 0).mean()),
        "avg_return": float(returns.mean()),
        "max_drawdown": float(drawdown.min()),
        "underperform_years": int(sum(value < 0 for value in annual)),
    }


def meta_score(metrics: dict[str, object]) -> float:
    return (
        math.log(max(float(metrics["final_cash"]), 1.0) / 1000.0)
        + 0.30 * float(metrics["win_rate"])
        + 0.50 * float(metrics["max_drawdown"])
        - 0.03 * int(metrics["underperform_years"])
    )


def test_methods(
    engine: ShadowResearchEngine,
    yearly: pd.DataFrame,
    trades: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    choices: dict[str, pd.DataFrame] = {}
    accounts: dict[str, pd.DataFrame] = {}
    for windows in WINDOWS:
        for score_kind in SCORE_KINDS:
            base_selected = engine.select_params(
                yearly,
                trades,
                window_years=list(windows),
                score_kind=score_kind,
                test_years=TEST_YEARS,
            )
            for cash_gate in CASH_GATES:
                selected = base_selected.copy()
                selected.loc[selected["score"] < cash_gate, "param_id"] = "CASH"
                method_id = (
                    f"w{'-'.join(str(value) for value in windows)}_"
                    f"{score_kind}_g{cash_gate:+.2f}"
                )
                account = selected_account(engine, selected, trades)
                dev = segment_metrics(account, DEV_YEARS)
                holdout = segment_metrics(account, HOLDOUT_YEARS)
                all_years = segment_metrics(
                    account, DEV_YEARS | HOLDOUT_YEARS
                )
                rows.append(
                    {
                        "method_id": method_id,
                        "windows": "-".join(str(value) for value in windows),
                        "score_kind": score_kind,
                        "cash_gate": cash_gate,
                        "dev_score": meta_score(dev),
                        **{f"dev_{key}": value for key, value in dev.items()},
                        **{
                            f"holdout_{key}": value
                            for key, value in holdout.items()
                        },
                        **{f"all_{key}": value for key, value in all_years.items()},
                    }
                )
                choices[method_id] = selected
                accounts[method_id] = account
    table = pd.DataFrame(rows).sort_values(
        ["dev_score", "dev_final_cash", "dev_win_rate"],
        ascending=False,
    )
    return table, choices, accounts


def baseline_metrics(line: str) -> dict[str, object]:
    path = (
        ROOT / "C" / "fit" / "fallback" / "final_trades.csv"
        if line == "C"
        else ROOT / "S" / "fit" / "selector" / "final_trades.csv"
    )
    frame = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    return {
        "path": str(path),
        "all": segment_metrics(frame, DEV_YEARS | HOLDOUT_YEARS),
        "development": segment_metrics(frame, DEV_YEARS),
        "holdout": segment_metrics(frame, HOLDOUT_YEARS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--line", choices=("C", "S"), required=True)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    engine = ShadowResearchEngine()
    params = c_params(engine) if args.line == "C" else s_params(engine)
    out = ROOT / args.line / "fit" / "strengthen"
    out.mkdir(parents=True, exist_ok=True)
    trades, yearly = generate_candidates(engine, params)
    methods, choices, accounts = test_methods(engine, yearly, trades)
    winner = methods.iloc[0]
    winner_id = str(winner["method_id"])
    winner_choice = choices[winner_id]
    winner_account = accounts[winner_id]
    baseline = baseline_metrics(args.line)

    pd.DataFrame([asdict(param) | {"param_id": param.param_id} for param in params]).to_csv(
        out / "candidate_params.csv", index=False, encoding="utf-8-sig"
    )
    methods.to_csv(out / "method_grid.csv", index=False, encoding="utf-8-sig")
    winner_choice.to_csv(
        out / "winner_selected_by_year.csv", index=False, encoding="utf-8-sig"
    )
    winner_account.to_csv(
        out / "winner_trades.csv", index=False, encoding="utf-8-sig"
    )

    holdout_gain = float(winner["holdout_final_cash"]) - float(
        baseline["holdout"]["final_cash"]
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "line": args.line,
        "candidate_params": len(params),
        "selector_methods": len(methods),
        "development_years": "2016-2023",
        "holdout_years": "2024-2026",
        "baseline": baseline,
        "winner": winner.to_dict(),
        "holdout_gain_vs_baseline": holdout_gain,
        "promotion_status": (
            "candidate"
            if holdout_gain > 0
            and float(winner["holdout_win_rate"])
            >= float(baseline["holdout"]["win_rate"])
            and float(winner["holdout_max_drawdown"])
            >= float(baseline["holdout"]["max_drawdown"])
            else "no_upgrade"
        ),
        "warning": (
            "The method is selected on 2016-2023 only. 2024-2026 is reported "
            "once as holdout. Results are gross and do not replace live config."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
