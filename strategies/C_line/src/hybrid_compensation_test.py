from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OUT = ROOT / "fit" / "hybrid_test"
C_TRADES = ROOT / "fit" / "fallback" / "final_trades.csv"
S_TRADES = WORKSPACE / "S" / "fit" / "selector" / "final_trades.csv"
DEV_YEARS = set(range(2016, 2024))
HOLDOUT_YEARS = {2024, 2025, 2026}
INITIAL_CASH = 10_000.0
COMMISSION_RATE = 0.0003
MIN_COMMISSION = 5.0


def attach_prices(frame: pd.DataFrame, preferred_root: Path) -> pd.DataFrame:
    result = frame.copy()
    if "entry_close" not in result.columns:
        result["entry_close"] = np.nan
    if "exit_close" not in result.columns:
        result["exit_close"] = np.nan
    for symbol, indexes in result.groupby("symbol").groups.items():
        path = preferred_root / "raw" / "etf" / f"{str(symbol).zfill(6)}.csv"
        if not path.exists():
            alternate = (
                WORKSPACE / "S" if preferred_root == ROOT else ROOT
            )
            path = alternate / "raw" / "etf" / f"{str(symbol).zfill(6)}.csv"
        prices = pd.read_csv(path, encoding="utf-8-sig")
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
        lookup = prices.dropna(subset=["date", "close"]).set_index("date")["close"]
        for index in indexes:
            entry_date = pd.Timestamp(result.at[index, "entry_date"])
            exit_date = pd.Timestamp(result.at[index, "exit_date"])
            result.at[index, "entry_close"] = lookup.get(
                entry_date, result.at[index, "entry_close"]
            )
            result.at[index, "exit_close"] = lookup.get(
                exit_date, result.at[index, "exit_close"]
            )
    return result


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    c = pd.read_csv(C_TRADES, dtype={"symbol": str}, encoding="utf-8-sig")
    s = pd.read_csv(S_TRADES, dtype={"symbol": str}, encoding="utf-8-sig")
    for frame in (c, s):
        frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
        frame["entry_date"] = pd.to_datetime(frame["entry_date"])
        frame["exit_date"] = pd.to_datetime(frame["exit_date"])
        frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    main_label = str(c["source"].value_counts().idxmax())
    main = c[c["source"].eq(main_label)].copy()
    c_fallback = c[~c["source"].eq(main_label)].copy()
    s_fallback = s[~s["source"].eq(main_label)].copy()
    s_main = s[s["source"].eq(main_label)].copy()
    main["family"] = "MAIN"
    c_fallback["family"] = "C"
    s_fallback["family"] = "S"
    s_formal = pd.concat(
        [
            attach_prices(s_main, ROOT),
            attach_prices(s_fallback, WORKSPACE / "S"),
        ],
        ignore_index=True,
        sort=False,
    ).sort_values(["entry_date", "symbol"])
    return (
        attach_prices(main, ROOT),
        attach_prices(c_fallback, ROOT),
        attach_prices(s_fallback, WORKSPACE / "S"),
        attach_prices(c, ROOT),
        s_formal,
    )


def historical_score(
    history: pd.DataFrame,
    family: str,
    date: pd.Timestamp,
    lookback: int | None,
    positive_only: bool,
) -> float:
    past = history[
        history["family"].eq(family)
        & history["exit_date"].lt(date)
    ].sort_values("exit_date")
    if lookback is not None:
        past = past.tail(lookback)
    returns = pd.to_numeric(past["ret"], errors="coerce").dropna()
    if len(returns) < 3:
        return 0.0
    mean = float(returns.mean())
    win = float((returns > 0).mean())
    downside = float(np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)))
    score = mean + 0.10 * (win - 0.5) - 0.25 * downside
    if positive_only and mean <= 0:
        return -np.inf
    return score


def build_account(
    main: pd.DataFrame,
    c_fallback: pd.DataFrame,
    s_fallback: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    candidates = pd.concat(
        [main, c_fallback, s_fallback], ignore_index=True, sort=False
    ).sort_values(["entry_date", "symbol"])
    history = pd.concat(
        [c_fallback, s_fallback], ignore_index=True, sort=False
    )
    rows: list[dict[str, object]] = []
    busy_until = pd.Timestamp.min
    for date, daily in candidates.groupby("entry_date", sort=True):
        date = pd.Timestamp(date)
        if date < busy_until:
            continue
        if daily["family"].eq("MAIN").any():
            ordered = daily.assign(
                meta_priority=np.where(daily["family"].eq("MAIN"), 1e9, -1e9)
            ).sort_values(
                ["meta_priority", "priority", "symbol"],
                ascending=[False, False, True],
            )
        else:
            if mode == "C_first":
                family_score = {"C": 2.0, "S": 1.0}
            elif mode == "S_first":
                family_score = {"C": 1.0, "S": 2.0}
            else:
                lookback = {
                    "past_3": 3,
                    "past_5": 5,
                    "past_10": 10,
                    "past_all": None,
                    "positive_5": 5,
                    "positive_all": None,
                }[mode]
                positive = mode.startswith("positive")
                family_score = {
                    family: historical_score(
                        history, family, date, lookback, positive
                    )
                    for family in ("C", "S")
                }
            ordered = daily.assign(
                meta_priority=daily["family"].map(family_score).fillna(-np.inf)
            ).sort_values(
                ["meta_priority", "priority", "symbol"],
                ascending=[False, False, True],
            )
        chosen = ordered.iloc[0]
        if not np.isfinite(float(chosen["meta_priority"])):
            continue
        item = chosen.to_dict()
        item["hybrid_mode"] = mode
        item["meta_priority"] = float(chosen["meta_priority"])
        rows.append(item)
        busy_until = pd.Timestamp(chosen["exit_date"])
    return pd.DataFrame(rows)


def metrics(frame: pd.DataFrame, years: set[int]) -> dict[str, float | int]:
    local = frame[pd.to_datetime(frame["entry_date"]).dt.year.isin(years)]
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
        local.assign(
            year=pd.to_datetime(local["entry_date"]).dt.year,
            ret_num=pd.to_numeric(local["ret"], errors="coerce"),
        )
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


def yearly_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    years = pd.to_datetime(frame["entry_date"]).dt.year
    rows: list[dict[str, float | int]] = []
    for year in sorted(years.dropna().unique()):
        stat = metrics(frame, {int(year)})
        rows.append({"year": int(year), **stat})
    return pd.DataFrame(rows)


def selection_score(metric: dict[str, float | int]) -> float:
    return (
        math.log(max(float(metric["final_1000"]), 1.0) / 1000.0)
        + 0.25 * float(np.nan_to_num(metric["win_rate"], nan=0.0))
        + 0.70 * float(metric["max_drawdown"])
        - 0.04 * int(metric["losing_years"])
    )


def rolling_select(
    accounts: dict[str, pd.DataFrame],
    window: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    choices: list[dict[str, object]] = []
    parts: list[pd.DataFrame] = []
    for year in range(2016, 2027):
        start = 2014 if window is None else max(2014, year - window)
        train_years = set(range(start, year))
        ranked = []
        for mode, account in accounts.items():
            stat = metrics(account, train_years)
            if int(stat["trades"]) < max(3, len(train_years)):
                continue
            ranked.append((selection_score(stat), mode))
        chosen = (
            max(ranked, key=lambda value: (value[0], value[1]))[1]
            if ranked
            else "C_only"
        )
        frame = accounts[chosen]
        years = pd.to_datetime(frame["entry_date"]).dt.year
        parts.append(frame[years.eq(year)].copy())
        choices.append(
            {
                "test_year": year,
                "train_start": start,
                "train_end": year - 1,
                "window": "all" if window is None else window,
                "hybrid_mode": chosen,
            }
        )
    pre = accounts["C_only"]
    pre_years = pd.to_datetime(pre["entry_date"]).dt.year
    combined = pd.concat(
        [pre[pre_years.lt(2016)], *parts], ignore_index=True, sort=False
    ).sort_values(["entry_date", "symbol"])
    kept: list[pd.Series] = []
    busy = pd.Timestamp.min
    for _, row in combined.iterrows():
        if pd.Timestamp(row["entry_date"]) < busy:
            continue
        kept.append(row)
        busy = pd.Timestamp(row["exit_date"])
    return pd.DataFrame(choices), pd.DataFrame(kept)


def commission(notional: float) -> float:
    return max(MIN_COMMISSION, abs(notional) * COMMISSION_RATE)


def net_account(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    cash = INITIAL_CASH
    rows: list[dict[str, object]] = []
    for _, trade in frame.sort_values(["entry_date", "symbol"]).iterrows():
        entry = float(trade["entry_close"])
        exit_price = float(trade["exit_close"])
        quantity = math.floor(cash / (entry * 100)) * 100
        while quantity >= 100 and quantity * entry + commission(
            quantity * entry
        ) > cash:
            quantity -= 100
        if quantity < 100:
            continue
        buy_value = quantity * entry
        buy_fee = commission(buy_value)
        sell_value = quantity * exit_price
        sell_fee = commission(sell_value)
        cash += sell_value - sell_fee - buy_value - buy_fee
        item = trade.to_dict()
        item.update(
            {
                "quantity": quantity,
                "buy_fee": buy_fee,
                "sell_fee": sell_fee,
                "net_cash": cash,
            }
        )
        rows.append(item)
    return pd.DataFrame(rows), {
        "initial_cash": INITIAL_CASH,
        "final_cash": float(cash),
        "net_profit": float(cash - INITIAL_CASH),
        "trades": len(rows),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (
        main_trades,
        c_fallback,
        s_fallback,
        c_formal,
        s_formal,
    ) = load_inputs()
    accounts = {
        "C_only": c_formal,
        "S_only": s_formal,
        **{
            mode: build_account(main_trades, c_fallback, s_fallback, mode)
            for mode in (
                "C_first",
                "S_first",
                "past_3",
                "past_5",
                "past_10",
                "past_all",
                "positive_5",
                "positive_all",
            )
        },
    }
    fixed_rows = []
    for mode, account in accounts.items():
        dev = metrics(account, DEV_YEARS)
        holdout = metrics(account, HOLDOUT_YEARS)
        fixed_rows.append(
            {
                "hybrid_mode": mode,
                "dev_score": selection_score(dev),
                **{f"dev_{key}": value for key, value in dev.items()},
                **{f"holdout_{key}": value for key, value in holdout.items()},
            }
        )
    fixed = pd.DataFrame(fixed_rows).sort_values(
        ["dev_score", "dev_final_1000"], ascending=False
    )

    method_rows = []
    choices_map: dict[str, pd.DataFrame] = {}
    logs_map: dict[str, pd.DataFrame] = {}
    for window in (3, 5, 7, None):
        method = f"w{window if window is not None else 'all'}"
        choices, log = rolling_select(accounts, window)
        choices_map[method] = choices
        logs_map[method] = log
        dev = metrics(log, DEV_YEARS)
        holdout = metrics(log, HOLDOUT_YEARS)
        method_rows.append(
            {
                "method": method,
                "dev_score": selection_score(dev),
                **{f"dev_{key}": value for key, value in dev.items()},
                **{f"holdout_{key}": value for key, value in holdout.items()},
            }
        )
    methods = pd.DataFrame(method_rows).sort_values(
        ["dev_score", "dev_final_1000"], ascending=False
    )
    winner = str(methods.iloc[0]["method"])
    winner_holdout = metrics(logs_map[winner], HOLDOUT_YEARS)
    c_holdout = metrics(c_formal, HOLDOUT_YEARS)
    s_holdout = metrics(s_formal, HOLDOUT_YEARS)
    best_baseline = (
        s_holdout
        if float(s_holdout["final_1000"]) >= float(c_holdout["final_1000"])
        else c_holdout
    )
    pass_count = int(
        (
            (methods["holdout_final_1000"] >= best_baseline["final_1000"])
            & (
                methods["holdout_win_rate"]
                >= float(best_baseline["win_rate"]) - 0.02
            )
            & (
                methods["holdout_max_drawdown"]
                >= float(best_baseline["max_drawdown"]) - 0.02
            )
        ).sum()
    )
    promoted = bool(
        winner_holdout["final_1000"] > best_baseline["final_1000"] * 1.05
        and winner_holdout["win_rate"] >= best_baseline["win_rate"] - 0.02
        and winner_holdout["max_drawdown"]
        >= best_baseline["max_drawdown"] - 0.02
        and int(winner_holdout["trades"]) >= 10
        and pass_count >= 2
    )
    net_log, net_summary = net_account(logs_map[winner])
    c_net_log, c_net_summary = net_account(c_formal)
    s_net_log, s_net_summary = net_account(s_formal)

    fixed.to_csv(OUT / "variants.csv", index=False, encoding="utf-8-sig")
    methods.to_csv(OUT / "rolling_methods.csv", index=False, encoding="utf-8-sig")
    choices_map[winner].to_csv(
        OUT / "winner_choices.csv", index=False, encoding="utf-8-sig"
    )
    logs_map[winner].to_csv(
        OUT / "winner_trades.csv", index=False, encoding="utf-8-sig"
    )
    net_log.to_csv(
        OUT / "winner_net_trades.csv", index=False, encoding="utf-8-sig"
    )
    yearly_metrics(logs_map[winner]).to_csv(
        OUT / "winner_yearly.csv", index=False, encoding="utf-8-sig"
    )
    c_net_log.to_csv(
        OUT / "c_baseline_net_trades.csv", index=False, encoding="utf-8-sig"
    )
    s_net_log.to_csv(
        OUT / "s_baseline_net_trades.csv", index=False, encoding="utf-8-sig"
    )
    source_contribution = (
        logs_map[winner]
        .assign(ret_num=pd.to_numeric(logs_map[winner]["ret"], errors="coerce"))
        .groupby(["family", "source"], dropna=False)
        .agg(
            trades=("ret_num", "size"),
            wins=("ret_num", lambda values: int((values > 0).sum())),
            avg_return=("ret_num", "mean"),
            compound_return=(
                "ret_num",
                lambda values: float(np.prod(1.0 + values.dropna()) - 1.0),
            ),
        )
        .reset_index()
    )
    source_contribution.to_csv(
        OUT / "source_contribution.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": "C/S causal compensation meta-selector",
        "main_strategy_unchanged": True,
        "c_baseline_holdout": c_holdout,
        "s_baseline_holdout": s_holdout,
        "development_selected_method": winner,
        "winner_holdout": winner_holdout,
        "winner_net": net_summary,
        "c_baseline_net": c_net_summary,
        "s_baseline_net": s_net_summary,
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
