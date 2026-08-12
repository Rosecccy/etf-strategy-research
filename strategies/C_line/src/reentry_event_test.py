from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "reentry_test"
BASELINE_PATH = ROOT / "fit" / "fallback" / "final_trades.csv"
ETF_DIR = ROOT / "raw" / "etf"
DEV_YEARS = set(range(2016, 2024))
HOLDOUT_YEARS = {2024, 2025, 2026}


@dataclass(frozen=True)
class Param:
    delay: int
    hold: int
    ma: int
    ret20_min: float
    min_pullback: float
    max_pullback: float
    reversal: str

    @property
    def key(self) -> str:
        return (
            f"d{self.delay}_h{self.hold}_ma{self.ma}_"
            f"r{int(round(self.ret20_min * 100)):+03d}_"
            f"pb{int(round(self.min_pullback * 100)):02d}-"
            f"{int(round(self.max_pullback * 100)):02d}_{self.reversal}"
        )


def params() -> list[Param]:
    values = []
    for item in product(
        (0, 2, 5),
        (10, 20, 40, 60),
        (20, 60, 120),
        (-0.03, 0.0, 0.03, 0.06),
        (0.0, 0.02, 0.04),
        (0.06, 0.10, 0.15),
        ("none", "ret1", "ret3"),
    ):
        param = Param(*item)
        if param.min_pullback <= param.max_pullback:
            values.append(param)
    return values


def load_baseline() -> pd.DataFrame:
    frame = pd.read_csv(
        BASELINE_PATH,
        dtype={"symbol": str},
        encoding="utf-8-sig",
    )
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["entry_date"] = pd.to_datetime(frame["entry_date"])
    frame["exit_date"] = pd.to_datetime(frame["exit_date"])
    frame["ret"] = pd.to_numeric(frame["ret"])
    frame["family"] = "BASE"
    return frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def load_prices(symbol: str) -> pd.DataFrame:
    frame = pd.read_csv(
        ETF_DIR / f"{symbol}.csv",
        encoding="utf-8-sig",
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = (
        frame.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    frame["ret1"] = frame["close"].pct_change()
    frame["ret3"] = frame["close"].pct_change(3)
    frame["ret20"] = frame["close"].pct_change(20)
    frame["high20"] = frame["close"].rolling(20, min_periods=10).max()
    frame["drawdown20"] = frame["close"] / frame["high20"] - 1.0
    for window in (20, 60, 120):
        frame[f"ma{window}"] = frame["close"].rolling(
            window, min_periods=max(10, window // 2)
        ).mean()
    return frame


def hit(row: pd.Series, param: Param) -> bool:
    close = float(row["close"])
    ma = float(row[f"ma{param.ma}"])
    ret20 = float(row["ret20"])
    drawdown = float(row["drawdown20"])
    if not all(np.isfinite(value) for value in (close, ma, ret20, drawdown)):
        return False
    if close <= ma or ret20 < param.ret20_min:
        return False
    if drawdown > -param.min_pullback or drawdown < -param.max_pullback:
        return False
    if param.reversal == "ret1":
        return bool(pd.notna(row["ret1"]) and float(row["ret1"]) > 0)
    if param.reversal == "ret3":
        return bool(pd.notna(row["ret3"]) and float(row["ret3"]) > 0)
    return True


def reentries(
    baseline: pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    param: Param,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordered = baseline.sort_values(["entry_date", "symbol"]).reset_index(
        drop=True
    )
    for index, trade in ordered.iterrows():
        symbol = str(trade["symbol"])
        prices = price_map[symbol]
        dates = prices["date"].to_numpy()
        exit_date = pd.Timestamp(trade["exit_date"])
        next_base_entry = (
            pd.Timestamp(ordered.iloc[index + 1]["entry_date"])
            if index + 1 < len(ordered)
            else pd.Timestamp(prices.iloc[-1]["date"]) + pd.Timedelta(days=1)
        )
        exit_pos = int(np.searchsorted(dates, np.datetime64(exit_date)))
        next_entry_pos = int(
            np.searchsorted(dates, np.datetime64(next_base_entry))
        )
        decision_start = exit_pos + param.delay
        last_decision = min(next_entry_pos - 1, len(prices) - 2)
        if decision_start > last_decision:
            continue
        decision_pos = None
        for pos in range(decision_start, last_decision + 1):
            if hit(prices.iloc[pos], param):
                decision_pos = pos
                break
        if decision_pos is None:
            continue
        entry_pos = decision_pos + 1
        planned_exit_pos = min(entry_pos + param.hold, len(prices) - 1)
        actual_exit_pos = min(planned_exit_pos, next_entry_pos)
        if actual_exit_pos <= entry_pos:
            continue
        entry = prices.iloc[entry_pos]
        exit_ = prices.iloc[actual_exit_pos]
        rows.append(
            {
                "source": "同标的趋势再入场",
                "family": "REENTRY",
                "symbol": symbol,
                "display_name": trade.get("display_name", symbol),
                "category": trade.get("category", ""),
                "decision_date": prices.iloc[decision_pos]["date"],
                "entry_date": entry["date"],
                "exit_date": exit_["date"],
                "entry_close": float(entry["close"]),
                "exit_close": float(exit_["close"]),
                "ret": float(exit_["close"] / entry["close"] - 1.0),
                "priority": 200.0,
                "param_id": param.key,
                "exit_reason": (
                    "main_signal_priority"
                    if actual_exit_pos == next_entry_pos
                    else "reentry_max_hold"
                ),
            }
        )
    return pd.DataFrame(rows)


def reentry_signals(
    baseline: pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    param: Param,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ordered = baseline.sort_values(["entry_date", "symbol"]).reset_index(
        drop=True
    )
    for index, trade in ordered.iterrows():
        symbol = str(trade["symbol"])
        prices = price_map[symbol]
        dates = prices["date"].to_numpy()
        exit_date = pd.Timestamp(trade["exit_date"])
        next_base_entry = (
            pd.Timestamp(ordered.iloc[index + 1]["entry_date"])
            if index + 1 < len(ordered)
            else pd.Timestamp(prices.iloc[-1]["date"]) + pd.Timedelta(days=1)
        )
        exit_pos = int(np.searchsorted(dates, np.datetime64(exit_date)))
        next_entry_pos = int(
            np.searchsorted(dates, np.datetime64(next_base_entry))
        )
        decision_start = exit_pos + param.delay
        last_decision = min(next_entry_pos - 1, len(prices) - 2)
        if decision_start > last_decision:
            continue
        for position in range(decision_start, last_decision + 1):
            if hit(prices.iloc[position], param):
                rows.append(
                    {
                        "trade": trade,
                        "symbol": symbol,
                        "decision_pos": position,
                        "entry_pos": position + 1,
                        "next_base_entry_pos": next_entry_pos,
                    }
                )
                break
    return rows


def materialize_reentries(
    signals: list[dict[str, object]],
    price_map: dict[str, pd.DataFrame],
    param: Param,
) -> pd.DataFrame:
    rows = []
    for signal in signals:
        symbol = str(signal["symbol"])
        prices = price_map[symbol]
        decision_pos = int(signal["decision_pos"])
        entry_pos = int(signal["entry_pos"])
        next_base_entry_pos = int(signal["next_base_entry_pos"])
        actual_exit_pos = min(
            entry_pos + param.hold,
            next_base_entry_pos,
            len(prices) - 1,
        )
        if actual_exit_pos <= entry_pos:
            continue
        trade = signal["trade"]
        entry = prices.iloc[entry_pos]
        exit_ = prices.iloc[actual_exit_pos]
        rows.append(
            {
                "source": "同标的趋势再入场",
                "family": "REENTRY",
                "symbol": symbol,
                "display_name": trade.get("display_name", symbol),
                "category": trade.get("category", ""),
                "decision_date": prices.iloc[decision_pos]["date"],
                "entry_date": entry["date"],
                "exit_date": exit_["date"],
                "entry_close": float(entry["close"]),
                "exit_close": float(exit_["close"]),
                "ret": float(exit_["close"] / entry["close"] - 1.0),
                "priority": 200.0,
                "param_id": param.key,
                "exit_reason": (
                    "main_signal_priority"
                    if actual_exit_pos == next_base_entry_pos
                    else "reentry_max_hold"
                ),
            }
        )
    return pd.DataFrame(rows)


def account(
    baseline: pd.DataFrame,
    extras: pd.DataFrame,
) -> pd.DataFrame:
    combined = pd.concat(
        [baseline, extras], ignore_index=True, sort=False
    ).sort_values(
        ["entry_date", "priority", "symbol"],
        ascending=[True, False, True],
    )
    rows = []
    busy = pd.Timestamp.min
    for _, trade in combined.iterrows():
        entry = pd.Timestamp(trade["entry_date"])
        if entry < busy:
            continue
        rows.append(trade)
        busy = pd.Timestamp(trade["exit_date"])
    return pd.DataFrame(rows)


def annual_summary(frame: pd.DataFrame, param_id: str) -> pd.DataFrame:
    rows = []
    local = frame.assign(year=frame["entry_date"].dt.year)
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


def metrics(frame: pd.DataFrame, years: set[int]) -> dict[str, float | int]:
    local = frame[frame["entry_date"].dt.year.isin(years)]
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
        local.assign(year=local["entry_date"].dt.year)
        .groupby("year")["ret"]
        .apply(lambda values: float(np.prod(1.0 + values) - 1.0))
    )
    return {
        "trades": int(len(returns)),
        "final_1000": float(equity.iloc[-1]),
        "win_rate": float((returns > 0).mean()),
        "max_drawdown": float(drawdown.min()),
        "losing_years": int((annual < 0).sum()),
    }


def score(stat: dict[str, float | int], kind: str) -> float:
    profit = math.log(max(float(stat["final_1000"]), 1.0) / 1000.0)
    win = float(np.nan_to_num(stat["win_rate"], nan=0.0))
    drawdown = float(stat["max_drawdown"])
    losing = int(stat["losing_years"])
    if kind == "profit":
        return profit
    if kind == "balanced":
        return profit + 0.25 * win + 0.70 * drawdown - 0.03 * losing
    return profit + 0.40 * win + 1.20 * drawdown - 0.06 * losing


def rolling_method(
    logs: dict[str, pd.DataFrame],
    window: int | None,
    kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    choices = []
    pieces = []
    for year in range(2016, 2027):
        start = 2014 if window is None else max(2014, year - window)
        train_years = set(range(start, year))
        ranked = []
        for key, log in logs.items():
            stat = metrics(log, train_years)
            if int(stat["trades"]) < max(3, len(train_years)):
                continue
            ranked.append((score(stat, kind), key))
        selected = (
            max(ranked, key=lambda value: (value[0], value[1]))[1]
            if ranked
            else "BASE"
        )
        selected_log = logs[selected]
        pieces.append(
            selected_log[selected_log["entry_date"].dt.year.eq(year)].copy()
        )
        choices.append(
            {
                "test_year": year,
                "train_start": start,
                "train_end": year - 1,
                "window": "all" if window is None else window,
                "score_kind": kind,
                "param_id": selected,
            }
        )
    pre = logs["BASE"]
    result = pd.concat(
        [pre[pre["entry_date"].dt.year.lt(2016)], *pieces],
        ignore_index=True,
        sort=False,
    ).sort_values(["entry_date", "symbol"])
    return pd.DataFrame(choices), result


def rolling_method_fast(
    logs: dict[str, pd.DataFrame],
    annual: pd.DataFrame,
    window: int | None,
    kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    choices = []
    pieces = []
    for year in range(2016, 2027):
        start = 2014 if window is None else max(2014, year - window)
        train = annual[
            annual["year"].between(start, year - 1)
        ].copy()
        grouped = (
            train.groupby("param_id")
            .agg(
                covered_years=("year", "nunique"),
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
        required_years = year - start
        grouped = grouped[
            (grouped["covered_years"] >= required_years)
            & (grouped["trades"] >= max(3, required_years))
        ].copy()
        if grouped.empty:
            selected = "BASE"
            selected_score = np.nan
        else:
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
                    + 0.40 * grouped["win_rate"]
                    + 1.20 * grouped["max_drawdown"]
                    - 0.06 * grouped["losing_years"]
                )
            picked = grouped.sort_values(
                ["selector_score", "log_growth", "win_rate", "param_id"],
                ascending=[False, False, False, True],
            ).iloc[0]
            selected = str(picked["param_id"])
            selected_score = float(picked["selector_score"])
        selected_log = logs[selected]
        pieces.append(
            selected_log[selected_log["entry_date"].dt.year.eq(year)].copy()
        )
        choices.append(
            {
                "test_year": year,
                "train_start": start,
                "train_end": year - 1,
                "window": "all" if window is None else window,
                "score_kind": kind,
                "param_id": selected,
                "train_score": selected_score,
            }
        )
    pre = logs["BASE"]
    result = pd.concat(
        [pre[pre["entry_date"].dt.year.lt(2016)], *pieces],
        ignore_index=True,
        sort=False,
    ).sort_values(["entry_date", "symbol"])
    return pd.DataFrame(choices), result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline()
    price_map = {
        symbol: load_prices(symbol)
        for symbol in baseline["symbol"].unique()
    }
    logs: dict[str, pd.DataFrame] = {"BASE": baseline}
    fixed_rows = []
    annual_parts = [annual_summary(baseline, "BASE")]
    condition_values = list(
        product(
            (0, 2, 5),
            (20, 60, 120),
            (-0.03, 0.0, 0.03, 0.06),
            (0.0, 0.02, 0.04),
            (0.06, 0.10, 0.15),
            ("none", "ret1", "ret3"),
        )
    )
    param_count = len(condition_values) * 4
    completed = 0
    for condition in condition_values:
        template = Param(
            delay=condition[0],
            hold=10,
            ma=condition[1],
            ret20_min=condition[2],
            min_pullback=condition[3],
            max_pullback=condition[4],
            reversal=condition[5],
        )
        signals = reentry_signals(baseline, price_map, template)
        for hold in (10, 20, 40, 60):
            param = Param(
                delay=template.delay,
                hold=hold,
                ma=template.ma,
                ret20_min=template.ret20_min,
                min_pullback=template.min_pullback,
                max_pullback=template.max_pullback,
                reversal=template.reversal,
            )
            log = account(
                baseline,
                materialize_reentries(signals, price_map, param),
            )
            logs[param.key] = log
            annual_parts.append(annual_summary(log, param.key))
            dev = metrics(log, DEV_YEARS)
            holdout = metrics(log, HOLDOUT_YEARS)
            fixed_rows.append(
                {
                    "param_id": param.key,
                    **param.__dict__,
                    **{f"dev_{key}": value for key, value in dev.items()},
                    **{
                        f"holdout_{key}": value
                        for key, value in holdout.items()
                    },
                }
            )
            completed += 1
            if completed % 500 == 0:
                print(f"[reentry {completed}/{param_count}]", flush=True)
    fixed = pd.DataFrame(fixed_rows)
    annual = pd.concat(annual_parts, ignore_index=True)

    method_rows = []
    choice_map = {}
    log_map = {}
    for window in (3, 5, 7, None):
        for kind in ("profit", "balanced", "stable"):
            method = f"w{window if window is not None else 'all'}_{kind}"
            choices, log = rolling_method_fast(
                logs, annual, window, kind
            )
            choice_map[method] = choices
            log_map[method] = log
            dev = metrics(log, DEV_YEARS)
            holdout = metrics(log, HOLDOUT_YEARS)
            method_rows.append(
                {
                    "method": method,
                    "dev_score": score(dev, kind),
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
    baseline_holdout = metrics(baseline, HOLDOUT_YEARS)
    passing = (
        (
            methods["holdout_final_1000"]
            >= float(baseline_holdout["final_1000"]) * 1.05
        )
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
    neighbor_count = int(passing.sum())
    promoted = bool(
        bool(passing.loc[winner.name]) and neighbor_count >= 2
    )

    fixed.to_csv(OUT / "fixed_grid.csv", index=False, encoding="utf-8-sig")
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
        "model": "same-symbol post-exit trend reentry",
        "candidate_params": param_count,
        "method_variants": len(methods),
        "baseline_holdout": baseline_holdout,
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
