from __future__ import annotations

import json
import math
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "hybrid_quality_gate_test"
TIMING = ROOT / "fit" / "execution_timing_audit"
DEV_YEARS = set(range(2016, 2024))
HOLDOUT_YEARS = {2024, 2025, 2026}


def read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["entry_date"] = pd.to_datetime(
        frame["entry_date"], errors="coerce"
    )
    frame["exit_date"] = pd.to_datetime(
        frame["exit_date"], errors="coerce"
    )
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    return frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def metrics(frame: pd.DataFrame, years: set[int]) -> dict[str, float | int]:
    local = frame[
        pd.to_datetime(frame["entry_date"]).dt.year.isin(years)
    ].copy()
    returns = pd.to_numeric(local["ret"], errors="coerce").dropna()
    if returns.empty:
        return {
            "trades": 0,
            "final_1000": 1000.0,
            "win_rate": np.nan,
            "max_drawdown": 0.0,
            "losing_years": 0,
        }
    equity = 1000.0 * (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    annual = (
        local.assign(
            year=pd.to_datetime(local["entry_date"]).dt.year,
            ret_num=pd.to_numeric(local["ret"], errors="coerce"),
        )
        .groupby("year")["ret_num"]
        .apply(lambda values: float(np.prod(1.0 + values) - 1.0))
    )
    return {
        "trades": int(len(returns)),
        "final_1000": float(equity.iloc[-1]),
        "win_rate": float((returns > 0).mean()),
        "max_drawdown": float(drawdown.min()),
        "losing_years": int((annual < 0).sum()),
    }


def selection_score(stat: dict[str, float | int], kind: str) -> float:
    profit = math.log(max(float(stat["final_1000"]), 1.0) / 1000.0)
    win = float(np.nan_to_num(stat["win_rate"], nan=0.0))
    drawdown = float(stat["max_drawdown"])
    losing = int(stat["losing_years"])
    if kind == "profit":
        return profit
    if kind == "balanced":
        return profit + 0.25 * win + 0.70 * drawdown - 0.03 * losing
    return profit + 0.45 * win + 1.20 * drawdown - 0.06 * losing


def gate_s_trades(
    s_fallback: pd.DataFrame,
    min_history: int,
    lookback: int | None,
    win_min: float,
    mean_min: float,
    score_min: float,
) -> pd.DataFrame:
    rows = []
    ordered = s_fallback.sort_values(["entry_date", "symbol"])
    for _, trade in ordered.iterrows():
        past = ordered[ordered["exit_date"].lt(trade["entry_date"])]
        if lookback is not None:
            past = past.sort_values("exit_date").tail(lookback)
        returns = pd.to_numeric(past["ret"], errors="coerce").dropna()
        count = int(len(returns))
        if count:
            win = float((returns > 0).mean())
            mean = float(returns.mean())
            downside = float(
                np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2))
            )
            quality = mean + 0.10 * (win - 0.5) - 0.25 * downside
        else:
            win = 0.5
            mean = 0.0
            quality = 0.0
        accepted = bool(
            count >= min_history
            and win >= win_min
            and mean >= mean_min
            and quality >= score_min
        )
        if accepted:
            item = trade.to_dict()
            item.update(
                {
                    "gate_history": count,
                    "gate_win": win,
                    "gate_mean": mean,
                    "gate_quality": quality,
                }
            )
            rows.append(item)
    return pd.DataFrame(rows)


def build_account(
    main: pd.DataFrame,
    c_fallback: pd.DataFrame,
    s_fallback: pd.DataFrame,
    param_id: str,
) -> pd.DataFrame:
    frames = []
    for family, priority, frame in (
        ("MAIN", 1_000_000.0, main),
        ("C", 200.0, c_fallback),
        ("S", 100.0, s_fallback),
    ):
        local = frame.copy()
        local["family"] = family
        local["priority"] = priority
        local["param_id"] = param_id
        frames.append(local)
    candidates = pd.concat(frames, ignore_index=True, sort=False).sort_values(
        ["entry_date", "priority", "symbol"],
        ascending=[True, False, True],
    )
    rows = []
    busy = pd.Timestamp.min
    for _, trade in candidates.iterrows():
        entry = pd.Timestamp(trade["entry_date"])
        if entry < busy:
            continue
        rows.append(trade)
        busy = pd.Timestamp(trade["exit_date"])
    return pd.DataFrame(rows).sort_values(["entry_date", "symbol"])


def annual_summary(frame: pd.DataFrame, param_id: str) -> pd.DataFrame:
    rows = []
    local = frame.assign(year=pd.to_datetime(frame["entry_date"]).dt.year)
    for year, group in local.groupby("year"):
        returns = pd.to_numeric(group["ret"], errors="coerce").dropna()
        equity = (1.0 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        rows.append(
            {
                "param_id": param_id,
                "year": int(year),
                "trades": int(len(returns)),
                "wins": int((returns > 0).sum()),
                "annual_return": float(equity.iloc[-1] - 1.0),
                "max_drawdown": float(drawdown.min()),
            }
        )
    return pd.DataFrame(rows)


def rolling_method(
    logs: dict[str, pd.DataFrame],
    annual: pd.DataFrame,
    window: int | None,
    kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    choices = []
    parts = []
    for year in range(2016, 2027):
        start = 2014 if window is None else max(2014, year - window)
        train = annual[annual["year"].between(start, year - 1)].copy()
        grouped = (
            train.groupby("param_id")
            .agg(
                covered=("year", "nunique"),
                trades=("trades", "sum"),
                wins=("wins", "sum"),
                log_growth=(
                    "annual_return",
                    lambda values: float(np.log1p(values).sum()),
                ),
                max_drawdown=("max_drawdown", "min"),
                losing_years=(
                    "annual_return",
                    lambda values: int((values < 0).sum()),
                ),
            )
            .reset_index()
        )
        required = year - start
        grouped = grouped[
            (grouped["covered"] >= required)
            & (grouped["trades"] >= max(3, required))
        ].copy()
        grouped["win_rate"] = grouped["wins"] / grouped["trades"]
        if kind == "profit":
            grouped["selector_score"] = grouped["log_growth"]
        elif kind == "balanced":
            grouped["selector_score"] = (
                grouped["log_growth"]
                + 0.25 * grouped["win_rate"]
                + 0.70 * grouped["max_drawdown"]
                - 0.03 * grouped["losing_years"]
            )
        else:
            grouped["selector_score"] = (
                grouped["log_growth"]
                + 0.45 * grouped["win_rate"]
                + 1.20 * grouped["max_drawdown"]
                - 0.06 * grouped["losing_years"]
            )
        picked = grouped.sort_values(
            ["selector_score", "log_growth", "win_rate", "param_id"],
            ascending=[False, False, False, True],
        ).iloc[0]
        selected = str(picked["param_id"])
        chosen = logs[selected]
        parts.append(
            chosen[
                pd.to_datetime(chosen["entry_date"]).dt.year.eq(year)
            ].copy()
        )
        choices.append(
            {
                "test_year": year,
                "train_start": start,
                "train_end": year - 1,
                "window": "all" if window is None else window,
                "score_kind": kind,
                "param_id": selected,
                "train_score": float(picked["selector_score"]),
            }
        )
    c_only = logs["C_ONLY"]
    result = pd.concat(
        [
            c_only[
                pd.to_datetime(c_only["entry_date"]).dt.year.lt(2016)
            ],
            *parts,
        ],
        ignore_index=True,
        sort=False,
    ).sort_values(["entry_date", "symbol"])
    return pd.DataFrame(choices), result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c = read(TIMING / "c_corrected_trades.csv")
    s = read(TIMING / "s_corrected_trades.csv")
    c_main = c[c["source"].astype(str).eq("主策略")].copy()
    c_fallback = c[~c["source"].astype(str).eq("主策略")].copy()
    s_fallback = s[~s["source"].astype(str).eq("主策略")].copy()

    logs: dict[str, pd.DataFrame] = {"C_ONLY": c}
    annual_parts = [annual_summary(c, "C_ONLY")]
    param_rows = [
        {
            "param_id": "C_ONLY",
            "min_history": np.nan,
            "lookback": np.nan,
            "win_min": np.nan,
            "mean_min": np.nan,
            "score_min": np.nan,
            "accepted_s_signals": 0,
        }
    ]
    seen: dict[tuple, str] = {}
    for values in product(
        (0, 3, 5, 10),
        (None, 5, 10, 20),
        (0.40, 0.50, 0.60, 0.70),
        (-0.01, 0.0, 0.005, 0.01, 0.02),
        (-0.02, 0.0, 0.01, 0.02),
    ):
        accepted = gate_s_trades(s_fallback, *values)
        signature = tuple(
            accepted[["symbol", "entry_date"]]
            .astype(str)
            .agg("|".join, axis=1)
            .tolist()
        ) if not accepted.empty else tuple()
        if signature in seen:
            continue
        param_id = (
            f"n{values[0]}_lb{values[1] or 'all'}_"
            f"w{int(values[2] * 100)}_m{int(values[3] * 1000)}_"
            f"q{int(values[4] * 1000)}"
        )
        seen[signature] = param_id
        log = build_account(c_main, c_fallback, accepted, param_id)
        logs[param_id] = log
        annual_parts.append(annual_summary(log, param_id))
        param_rows.append(
            {
                "param_id": param_id,
                "min_history": values[0],
                "lookback": values[1] or "all",
                "win_min": values[2],
                "mean_min": values[3],
                "score_min": values[4],
                "accepted_s_signals": int(len(accepted)),
                **{
                    f"dev_{key}": value
                    for key, value in metrics(log, DEV_YEARS).items()
                },
                **{
                    f"holdout_{key}": value
                    for key, value in metrics(log, HOLDOUT_YEARS).items()
                },
            }
        )
    annual = pd.concat(annual_parts, ignore_index=True)
    method_rows = []
    choice_map = {}
    log_map = {}
    for window in (3, 5, 7, None):
        for kind in ("profit", "balanced", "stable"):
            method = f"w{window if window is not None else 'all'}_{kind}"
            choices, log = rolling_method(logs, annual, window, kind)
            choice_map[method] = choices
            log_map[method] = log
            dev = metrics(log, DEV_YEARS)
            holdout = metrics(log, HOLDOUT_YEARS)
            method_rows.append(
                {
                    "method": method,
                    "dev_score": selection_score(dev, kind),
                    **{f"dev_{key}": value for key, value in dev.items()},
                    **{
                        f"holdout_{key}": value
                        for key, value in holdout.items()
                    },
                }
            )
    methods = pd.DataFrame(method_rows).sort_values(
        ["dev_score", "dev_final_1000"], ascending=False
    )
    winner = methods.iloc[0]
    winner_id = str(winner["method"])
    c_holdout = metrics(c, HOLDOUT_YEARS)
    s_holdout = metrics(s, HOLDOUT_YEARS)
    best = s_holdout if s_holdout["final_1000"] >= c_holdout["final_1000"] else c_holdout
    passing = (
        (methods["holdout_final_1000"] >= best["final_1000"] * 1.05)
        & (methods["holdout_win_rate"] >= best["win_rate"] - 0.02)
        & (
            methods["holdout_max_drawdown"]
            >= best["max_drawdown"] - 0.02
        )
        & (methods["holdout_trades"] >= 10)
    )
    neighbor_count = int(passing.sum())
    promoted = bool(passing.loc[winner.name] and neighbor_count >= 2)

    pd.DataFrame(param_rows).to_csv(
        OUT / "candidate_gates.csv", index=False, encoding="utf-8-sig"
    )
    methods.to_csv(
        OUT / "rolling_methods.csv", index=False, encoding="utf-8-sig"
    )
    choice_map[winner_id].to_csv(
        OUT / "winner_choices.csv", index=False, encoding="utf-8-sig"
    )
    log_map[winner_id].to_csv(
        OUT / "winner_trades.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": "execution-corrected hybrid with causal S quality gate",
        "unique_gate_signatures": len(logs) - 1,
        "method_variants": len(methods),
        "corrected_c_holdout": c_holdout,
        "corrected_s_holdout": s_holdout,
        "development_selected_method": winner_id,
        "winner": winner.to_dict(),
        "holdout_neighbors_passing": neighbor_count,
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
