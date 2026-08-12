from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from shadow_model import Param, ShadowResearchEngine


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "control_test"
SELECTED_PATH = ROOT / "fit" / "selector" / "selected_by_year.csv"
FORMAL_TRADES_PATH = ROOT / "fit" / "selector" / "final_trades.csv"
DEV_YEARS = set(range(2016, 2024))
HOLDOUT_YEARS = {2024, 2025, 2026}
MODES = (
    "flat",
    "category_top2",
    "category_top3",
    "category_consensus2",
    "category_stable2",
    "category_all_mean",
)


def parse_param(param_id: str) -> Param:
    pattern = re.compile(
        r"^(?P<scope>.+)_i(?P<idle>\d+)_h(?P<hold>\d+)_ma(?P<ma>\d+)"
        r"_m(?P<mom>\d+)_(?P<model>[^_]+)_a(?P<alpha>\d+)"
        r"_v(?P<vol>\d+)_ac(?P<accel>\d+)_hist(?P<history>\d+)"
        r"_amt(?P<amount>\d+)$"
    )
    match = pattern.match(param_id)
    if match is None:
        raise ValueError(f"Cannot parse param_id: {param_id}")
    values = match.groupdict()
    return Param(
        values["scope"],
        int(values["idle"]),
        int(values["hold"]),
        int(values["ma"]),
        int(values["mom"]),
        values["model"],
        int(values["alpha"]) / 100.0,
        int(values["vol"]) / 100.0,
        int(values["accel"]) / 100.0,
        int(values["history"]),
        float(values["amount"]),
    )


def choose_candidate(
    engine: ShadowResearchEngine,
    candidates: tuple[tuple[float, str, dict], ...],
    mode: str,
) -> tuple[float, str, dict] | None:
    if not candidates:
        return None
    if mode == "flat":
        return candidates[0]
    groups: dict[str, list[tuple[float, str, dict]]] = {}
    for candidate in candidates:
        category = str(engine.meta[candidate[1]]["category"])
        groups.setdefault(category, []).append(candidate)

    scored: list[tuple[float, str, tuple[float, str, dict]]] = []
    for category, items in groups.items():
        ordered = sorted(items, key=lambda item: (-item[0], item[1]))
        scores = np.array([item[0] for item in ordered], dtype=float)
        if mode == "category_top2":
            use = scores[:2]
            score = float(use.mean()) - (0.05 if len(use) < 2 else 0.0)
        elif mode == "category_top3":
            use = scores[:3]
            score = float(use.mean()) - 0.03 * max(0, 3 - len(use))
        elif mode == "category_consensus2":
            use = scores[:2]
            score = 0.55 * float(use.max()) + 0.45 * float(use.mean())
            score -= 0.05 if len(use) < 2 else 0.0
        elif mode == "category_stable2":
            use = scores[:2]
            score = float(use.mean()) - 0.35 * float(use.std(ddof=0))
            score -= 0.05 if len(use) < 2 else 0.0
        elif mode == "category_all_mean":
            score = 0.70 * float(scores.mean()) + 0.30 * float(scores.max())
        else:
            raise ValueError(f"Unknown mode: {mode}")
        scored.append((score, category, ordered[0]))
    return max(scored, key=lambda item: (item[0], item[1]))[2]


def generate_trades(
    engine: ShadowResearchEngine,
    param: Param,
    mode: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for start, end, idle_days in engine.idle_windows:
        if idle_days < param.idle:
            continue
        idle_calendar = engine.calendar[
            (engine.calendar >= start) & (engine.calendar <= end)
        ].reset_index(drop=True)
        if len(idle_calendar) < param.idle:
            continue
        current = pd.Timestamp(idle_calendar.iloc[param.idle - 1])
        while current <= end:
            candidates = engine.candidate_list(
                current.value,
                param.scope,
                param.ma,
                param.mom,
                param.model,
                param.alpha,
                param.min_history,
                param.min_amount20,
                param.vol_cap,
                param.accel_cap,
            )
            chosen = choose_candidate(engine, candidates, mode)
            if chosen is None:
                next_date = engine.next_date(current + pd.Timedelta(days=1))
                if next_date is None:
                    break
                current = next_date
                continue
            _, symbol, extra = chosen
            frame = engine.panel[symbol]
            entry_pos = engine.pos_after(
                frame, current + pd.Timedelta(days=1)
            )
            end_pos = engine.pos_before(frame, end)
            if entry_pos is None or end_pos is None or entry_pos > end_pos:
                break
            exit_pos = min(entry_pos + param.hold, end_pos)
            if exit_pos <= entry_pos:
                break
            entry = frame.iloc[entry_pos]
            exit_row = frame.iloc[exit_pos]
            rows.append(
                {
                    "source": "空仓补偿_S1",
                    "source_type": "空仓补偿",
                    "category": engine.meta[symbol]["category"],
                    "symbol": symbol,
                    "display_name": engine.meta[symbol]["display_name"],
                    "decision_date": current,
                    "entry_date": pd.Timestamp(entry["date"]),
                    "exit_date": pd.Timestamp(exit_row["date"]),
                    "entry_close": float(entry["close"]),
                    "exit_close": float(exit_row["close"]),
                    "ret": float(exit_row["close"] / entry["close"] - 1.0),
                    "priority": 100.0,
                    "ranking_mode": mode,
                    "scope": param.scope,
                    "param_id": param.param_id,
                    "idle_days": param.idle,
                    "hold": param.hold,
                    "ma": param.ma,
                    "mom": param.mom,
                    "model": param.model,
                    "alpha": param.alpha,
                    **extra,
                }
            )
            next_date = engine.next_date(
                pd.Timestamp(exit_row["date"]) + pd.Timedelta(days=1)
            )
            if next_date is None:
                break
            current = next_date
    return pd.DataFrame(rows)


def build_mode_account(
    engine: ShadowResearchEngine,
    annual: pd.DataFrame,
    generated: dict[tuple[str, str], pd.DataFrame],
    mode: str,
) -> pd.DataFrame:
    pieces = [engine.main.copy()]
    for _, choice in annual.iterrows():
        param_id = str(choice["param_id"])
        if param_id == "CASH":
            continue
        year = int(choice["year"])
        frame = generated[(mode, param_id)].copy()
        entry_year = pd.to_datetime(frame["entry_date"]).dt.year
        pieces.append(frame[entry_year.eq(year)].copy())
    return engine.account_from_trades(
        pd.concat(pieces, ignore_index=True, sort=False)
    )


def metrics(frame: pd.DataFrame, years: set[int]) -> dict[str, float | int]:
    local = frame.copy()
    local["year"] = pd.to_datetime(local["entry_date"]).dt.year
    local = local[local["year"].isin(years)].copy()
    returns = pd.to_numeric(local["ret"], errors="coerce").dropna()
    if returns.empty:
        return {
            "trades": 0,
            "final_1000": 1000.0,
            "win_rate": np.nan,
            "avg_return": np.nan,
            "max_drawdown": 0.0,
            "losing_years": 0,
        }
    equity = 1000.0 * (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    yearly = (
        local.assign(ret_num=pd.to_numeric(local["ret"], errors="coerce"))
        .groupby("year")["ret_num"]
        .apply(lambda values: float(np.prod(1.0 + values.dropna()) - 1.0))
    )
    return {
        "trades": int(len(returns)),
        "final_1000": float(equity.iloc[-1]),
        "win_rate": float((returns > 0).mean()),
        "avg_return": float(returns.mean()),
        "max_drawdown": float(drawdown.min()),
        "losing_years": int((yearly < 0).sum()),
    }


def score(metric: dict[str, float | int]) -> float:
    return (
        math.log(max(float(metric["final_1000"]), 1.0) / 1000.0)
        + 0.25 * float(np.nan_to_num(metric["win_rate"], nan=0.0))
        + 0.60 * float(metric["max_drawdown"])
        - 0.04 * int(metric["losing_years"])
    )


def rolling_mode(
    accounts: dict[str, pd.DataFrame],
    window: int | None,
    engine: ShadowResearchEngine,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    choices: list[dict[str, object]] = []
    pieces: list[pd.DataFrame] = []
    for year in range(2016, 2027):
        start = 2014 if window is None else max(2014, year - window)
        past = set(range(start, year))
        ranked = [
            (score(metrics(account, past)), mode)
            for mode, account in accounts.items()
            if int(metrics(account, past)["trades"]) >= max(3, len(past))
        ]
        chosen = (
            max(ranked, key=lambda item: (item[0], item[1]))[1]
            if ranked
            else "flat"
        )
        train_score = (
            max(ranked, key=lambda item: (item[0], item[1]))[0]
            if ranked
            else np.nan
        )
        selected = accounts[chosen].copy()
        selected_year = pd.to_datetime(selected["entry_date"]).dt.year
        pieces.append(selected[selected_year.eq(year)].copy())
        choices.append(
            {
                "test_year": year,
                "train_start": start,
                "train_end": year - 1,
                "window": "all" if window is None else window,
                "ranking_mode": chosen,
                "train_score": train_score,
            }
        )
    result = engine.account_from_trades(
        pd.concat(
            [
                accounts["flat"][
                    pd.to_datetime(accounts["flat"]["entry_date"]).dt.year
                    < 2016
                ],
                *pieces,
            ],
            ignore_index=True,
            sort=False,
        )
    )
    return pd.DataFrame(choices), result


def approximate_net(frame: pd.DataFrame) -> float:
    cash = 10_000.0
    for value in pd.to_numeric(frame["ret"], errors="coerce").dropna():
        buy_fee = max(5.0, cash * 0.0003)
        invested = max(0.0, cash - buy_fee)
        gross_exit = invested * (1.0 + float(value))
        sell_fee = max(5.0, gross_exit * 0.0003)
        cash = gross_exit - sell_fee
    return float(cash)


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
    generated: dict[tuple[str, str], pd.DataFrame] = {}
    total = len(MODES) * len(params)
    counter = 0
    for mode in MODES:
        for param_id, param in params.items():
            counter += 1
            generated[(mode, param_id)] = generate_trades(
                engine, param, mode
            )
            print(f"[S hierarchy {counter:02d}/{total}] {mode} {param_id}")

    accounts = {
        mode: build_mode_account(engine, annual, generated, mode)
        for mode in MODES
    }
    fixed_rows = []
    for mode, account in accounts.items():
        dev = metrics(account, DEV_YEARS)
        holdout = metrics(account, HOLDOUT_YEARS)
        fixed_rows.append(
            {
                "ranking_mode": mode,
                "dev_score": score(dev),
                **{f"dev_{key}": value for key, value in dev.items()},
                **{
                    f"holdout_{key}": value
                    for key, value in holdout.items()
                },
                "approx_net_final_10000": approximate_net(account),
            }
        )
    fixed = pd.DataFrame(fixed_rows).sort_values(
        ["dev_score", "dev_final_1000"], ascending=False
    )

    methods: list[dict[str, object]] = []
    choices_map: dict[str, pd.DataFrame] = {}
    logs_map: dict[str, pd.DataFrame] = {}
    for window in (3, 5, 7, None):
        method_id = f"w{window if window is not None else 'all'}"
        choices, log = rolling_mode(accounts, window, engine)
        choices_map[method_id] = choices
        logs_map[method_id] = log
        dev = metrics(log, DEV_YEARS)
        holdout = metrics(log, HOLDOUT_YEARS)
        methods.append(
            {
                "method": method_id,
                "dev_score": score(dev),
                **{f"dev_{key}": value for key, value in dev.items()},
                **{
                    f"holdout_{key}": value
                    for key, value in holdout.items()
                },
                "approx_net_final_10000": approximate_net(log),
            }
        )
    method_table = pd.DataFrame(methods).sort_values(
        ["dev_score", "dev_final_1000"], ascending=False
    )
    winner = str(method_table.iloc[0]["method"])

    formal = pd.read_csv(
        FORMAL_TRADES_PATH, dtype={"symbol": str}, encoding="utf-8-sig"
    )
    formal["entry_date"] = pd.to_datetime(formal["entry_date"])
    formal_baseline = metrics(formal, HOLDOUT_YEARS)
    reproduced = metrics(accounts["flat"], DEV_YEARS | HOLDOUT_YEARS)
    winner_holdout = metrics(logs_map[winner], HOLDOUT_YEARS)
    neighbor_passes = int(
        (
            (method_table["holdout_final_1000"] >= formal_baseline["final_1000"])
            & (
                method_table["holdout_win_rate"]
                >= float(formal_baseline["win_rate"]) - 0.02
            )
            & (
                method_table["holdout_max_drawdown"]
                >= float(formal_baseline["max_drawdown"]) - 0.02
            )
        ).sum()
    )
    promoted = bool(
        winner_holdout["final_1000"] > formal_baseline["final_1000"]
        and winner_holdout["win_rate"] >= formal_baseline["win_rate"]
        and winner_holdout["max_drawdown"] >= formal_baseline["max_drawdown"]
        and neighbor_passes >= 2
    )

    fixed.to_csv(
        OUT / "hierarchy_variants.csv", index=False, encoding="utf-8-sig"
    )
    method_table.to_csv(
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
        "control_variable": "etf_ranking_hierarchy_only",
        "formal_baseline_holdout": formal_baseline,
        "flat_reproduction_all": reproduced,
        "development_selected_method": winner,
        "winner_holdout": winner_holdout,
        "neighbor_methods_passing_holdout_floor": neighbor_passes,
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
