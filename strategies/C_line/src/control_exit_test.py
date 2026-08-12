from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "control_test"
TRADES_PATH = ROOT / "fit" / "fallback" / "final_trades.csv"
ETF_DIR = ROOT / "raw" / "etf"
DEV_YEARS = set(range(2016, 2024))
HOLDOUT_YEARS = {2024, 2025, 2026}
INITIAL_CASH = 10_000.0
COMMISSION_RATE = 0.0003
MIN_COMMISSION = 5.0


@dataclass(frozen=True)
class ExitRule:
    key: str
    family: str
    threshold: float = 0.0
    days: int = 0
    min_hold: int = 0


def commission(notional: float) -> float:
    return max(MIN_COMMISSION, abs(notional) * COMMISSION_RATE)


def load_prices(symbol: str) -> pd.DataFrame:
    frame = pd.read_csv(
        ETF_DIR / f"{symbol}.csv",
        dtype={"symbol": str},
        encoding="utf-8-sig",
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date")
    frame = frame.drop_duplicates("date", keep="last").reset_index(drop=True)
    frame["ret20"] = frame["close"].pct_change(20)
    frame["ma20"] = frame["close"].rolling(20, min_periods=15).mean()
    return frame


def rules() -> list[ExitRule]:
    result = [ExitRule("baseline", "baseline")]
    for threshold in (0.02, 0.04, 0.06, 0.10):
        for days in (2, 3, 5, 8, 10, 15):
            result.append(
                ExitRule(
                    f"strong_{int(threshold * 100):02d}_extend_{days}",
                    "strong_extend",
                    threshold,
                    days,
                )
            )
    for days in (3, 5, 8, 10, 15, 20):
        result.append(ExitRule(f"ma20_hold_{days}", "ma20_hold", days=days))
    for threshold in (-0.03, -0.05, -0.07):
        for drawdown in (0.05, 0.07, 0.10):
            result.append(
                ExitRule(
                    f"weak_{abs(int(threshold * 100)):02d}_dd"
                    f"{int(drawdown * 100):02d}",
                    "weak_break",
                    threshold,
                    int(round(drawdown * 100)),
                    15,
                )
            )
    for activation in (0.05, 0.10, 0.15):
        for drawdown in (0.04, 0.06, 0.08, 0.10):
            result.append(
                ExitRule(
                    f"trail_a{int(activation * 100):02d}_dd"
                    f"{int(drawdown * 100):02d}",
                    "trailing",
                    activation,
                    int(round(drawdown * 100)),
                    10,
                )
            )
    return result


def position_on_or_after(frame: pd.DataFrame, date: pd.Timestamp) -> int | None:
    pos = int(frame["date"].searchsorted(date, side="left"))
    return pos if pos < len(frame) else None


def adjusted_exit(
    trade: pd.Series,
    frame: pd.DataFrame,
    rule: ExitRule,
) -> pd.Timestamp:
    planned = pd.Timestamp(trade["exit_date"])
    if rule.family == "baseline" or str(trade["source"]) != "主策略":
        return planned

    entry_pos = position_on_or_after(frame, pd.Timestamp(trade["entry_date"]))
    exit_pos = position_on_or_after(frame, planned)
    if entry_pos is None or exit_pos is None or exit_pos <= entry_pos:
        return planned

    if rule.family == "strong_extend":
        decision_pos = max(entry_pos, exit_pos - 1)
        ret20 = frame.iloc[decision_pos]["ret20"]
        if pd.notna(ret20) and float(ret20) >= rule.threshold:
            return pd.Timestamp(
                frame.iloc[min(exit_pos + rule.days, len(frame) - 1)]["date"]
            )
        return planned

    if rule.family == "ma20_hold":
        decision_pos = max(entry_pos, exit_pos - 1)
        row = frame.iloc[decision_pos]
        if pd.isna(row["ma20"]) or float(row["close"]) < float(row["ma20"]):
            return planned
        last_signal = min(exit_pos + rule.days - 1, len(frame) - 2)
        for signal_pos in range(exit_pos, last_signal + 1):
            signal = frame.iloc[signal_pos]
            if (
                pd.notna(signal["ma20"])
                and float(signal["close"]) < float(signal["ma20"])
            ):
                return pd.Timestamp(frame.iloc[signal_pos + 1]["date"])
        return pd.Timestamp(
            frame.iloc[min(exit_pos + rule.days, len(frame) - 1)]["date"]
        )

    closes = frame["close"].to_numpy(dtype=float)
    end_signal = min(exit_pos - 1, len(frame) - 2)
    start_signal = min(entry_pos + rule.min_hold, end_signal)
    if start_signal > end_signal:
        return planned
    entry_close = closes[entry_pos]
    peak = entry_close
    for signal_pos in range(entry_pos, end_signal + 1):
        close = closes[signal_pos]
        peak = max(peak, close)
        if signal_pos < start_signal:
            continue
        drawdown = close / peak - 1.0
        if rule.family == "weak_break":
            ret20 = frame.iloc[signal_pos]["ret20"]
            dd_limit = -rule.days / 100.0
            if (
                pd.notna(ret20)
                and float(ret20) <= rule.threshold
                and drawdown <= dd_limit
            ):
                return pd.Timestamp(frame.iloc[signal_pos + 1]["date"])
        elif rule.family == "trailing":
            peak_return = peak / entry_close - 1.0
            dd_limit = -rule.days / 100.0
            if peak_return >= rule.threshold and drawdown <= dd_limit:
                return pd.Timestamp(frame.iloc[signal_pos + 1]["date"])
    return planned


def build_rule_log(
    source: pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    rule: ExitRule,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    busy_until = pd.Timestamp.min
    for _, trade in source.sort_values(["entry_date", "symbol"]).iterrows():
        entry_date = pd.Timestamp(trade["entry_date"])
        # The formal account allows a close-price sell and a new close-price
        # buy on the same trading date.
        if entry_date < busy_until:
            continue
        symbol = str(trade["symbol"]).zfill(6)
        frame = price_map[symbol]
        entry_pos = position_on_or_after(frame, entry_date)
        if entry_pos is None:
            continue
        exit_date = adjusted_exit(trade, frame, rule)
        exit_pos = position_on_or_after(frame, exit_date)
        if exit_pos is None or exit_pos <= entry_pos:
            continue
        entry_close = float(frame.iloc[entry_pos]["close"])
        exit_close = float(frame.iloc[exit_pos]["close"])
        ret = (
            float(trade["ret"])
            if rule.family == "baseline"
            else exit_close / entry_close - 1.0
        )
        item = trade.to_dict()
        item.update(
            {
                "rule": rule.key,
                "entry_date": entry_date,
                "original_exit_date": pd.Timestamp(trade["exit_date"]),
                "exit_date": pd.Timestamp(frame.iloc[exit_pos]["date"]),
                "entry_close_test": entry_close,
                "exit_close_test": exit_close,
                "ret_test": ret,
                "exit_shift_days": exit_pos
                - int(frame["date"].searchsorted(trade["exit_date"])),
            }
        )
        rows.append(item)
        busy_until = pd.Timestamp(item["exit_date"])
    return pd.DataFrame(rows)


def metrics(frame: pd.DataFrame, years: set[int]) -> dict[str, float | int]:
    if frame.empty:
        return {
            "trades": 0,
            "final_1000": 1000.0,
            "win_rate": np.nan,
            "avg_return": np.nan,
            "max_drawdown": 0.0,
            "losing_years": 0,
        }
    local = frame.copy()
    local["year"] = pd.to_datetime(local["entry_date"]).dt.year
    local = local[local["year"].isin(years)].copy()
    returns = pd.to_numeric(local["ret_test"], errors="coerce").dropna()
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
        local.assign(ret_num=pd.to_numeric(local["ret_test"], errors="coerce"))
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


def selection_score(metric: dict[str, float | int]) -> float:
    return (
        math.log(max(float(metric["final_1000"]), 1.0) / 1000.0)
        + 0.25 * float(np.nan_to_num(metric["win_rate"], nan=0.0))
        + 0.60 * float(metric["max_drawdown"])
        - 0.04 * int(metric["losing_years"])
    )


def rolling_choices(
    logs: dict[str, pd.DataFrame],
    window: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    choices: list[dict[str, object]] = []
    pieces: list[pd.DataFrame] = []
    for year in range(2016, 2027):
        start = 2014 if window is None else max(2014, year - window)
        past_years = set(range(start, year))
        ranked: list[tuple[float, str, dict[str, float | int]]] = []
        for key, log in logs.items():
            metric = metrics(log, past_years)
            if int(metric["trades"]) < max(3, len(past_years)):
                continue
            ranked.append((selection_score(metric), key, metric))
        if not ranked:
            chosen = "baseline"
            score = np.nan
        else:
            score, chosen, _ = max(ranked, key=lambda item: (item[0], item[1]))
        selected = logs[chosen].copy()
        selected["entry_year"] = pd.to_datetime(selected["entry_date"]).dt.year
        pieces.append(selected[selected["entry_year"].eq(year)].copy())
        choices.append(
            {
                "test_year": year,
                "train_start": start,
                "train_end": year - 1,
                "window": "all" if window is None else window,
                "rule": chosen,
                "train_score": score,
            }
        )
    combined = (
        pd.concat(pieces, ignore_index=True, sort=False)
        if pieces
        else pd.DataFrame()
    )
    pre_2016 = logs["baseline"].copy()
    pre_2016_year = pd.to_datetime(pre_2016["entry_date"]).dt.year
    combined = pd.concat(
        [pre_2016[pre_2016_year < 2016], combined],
        ignore_index=True,
        sort=False,
    )
    if not combined.empty:
        combined = combined.sort_values(["entry_date", "symbol"])
        kept: list[pd.Series] = []
        busy = pd.Timestamp.min
        for _, row in combined.iterrows():
            if pd.Timestamp(row["entry_date"]) < busy:
                continue
            kept.append(row)
            busy = pd.Timestamp(row["exit_date"])
        combined = pd.DataFrame(kept)
    return pd.DataFrame(choices), combined


def net_account(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    cash = INITIAL_CASH
    rows: list[dict[str, object]] = []
    for _, trade in frame.sort_values(["entry_date", "symbol"]).iterrows():
        price = float(trade["entry_close_test"])
        quantity = math.floor(cash / (price * 100)) * 100
        while quantity >= 100 and quantity * price + commission(
            quantity * price
        ) > cash:
            quantity -= 100
        if quantity < 100:
            continue
        buy_value = quantity * price
        buy_fee = commission(buy_value)
        cash -= buy_value + buy_fee
        sell_value = quantity * float(trade["exit_close_test"])
        sell_fee = commission(sell_value)
        cash += sell_value - sell_fee
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

    logs: dict[str, pd.DataFrame] = {}
    fixed_rows: list[dict[str, object]] = []
    for index, rule in enumerate(rules(), start=1):
        log = build_rule_log(source, price_map, rule)
        logs[rule.key] = log
        fixed_rows.append(
            {
                "rule": rule.key,
                "family": rule.family,
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
        print(f"[C exit {index:02d}/{len(rules())}] {rule.key}")
    fixed = pd.DataFrame(fixed_rows)
    fixed["dev_score"] = fixed.apply(
        lambda row: selection_score(
            {
                key.removeprefix("dev_"): row[key]
                for key in row.index
                if key.startswith("dev_")
            }
        ),
        axis=1,
    )
    fixed = fixed.sort_values(
        ["dev_score", "dev_final_1000", "dev_win_rate"], ascending=False
    )

    method_rows: list[dict[str, object]] = []
    method_choices: dict[str, pd.DataFrame] = {}
    method_logs: dict[str, pd.DataFrame] = {}
    for window in (3, 5, 7, None):
        choices, log = rolling_choices(logs, window)
        method_id = f"w{window if window is not None else 'all'}"
        method_choices[method_id] = choices
        method_logs[method_id] = log
        dev = metrics(log, DEV_YEARS)
        holdout = metrics(log, HOLDOUT_YEARS)
        method_rows.append(
            {
                "method": method_id,
                "dev_score": selection_score(dev),
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
    winner = str(methods.iloc[0]["method"])
    baseline = logs["baseline"]
    baseline_holdout = metrics(baseline, HOLDOUT_YEARS)
    winner_holdout = metrics(method_logs[winner], HOLDOUT_YEARS)
    neighbor_passes = int(
        (
            (methods["holdout_final_1000"] >= baseline_holdout["final_1000"])
            & (
                methods["holdout_win_rate"]
                >= float(baseline_holdout["win_rate"]) - 0.02
            )
            & (
                methods["holdout_max_drawdown"]
                >= float(baseline_holdout["max_drawdown"]) - 0.02
            )
        ).sum()
    )
    promoted = bool(
        winner_holdout["final_1000"] > baseline_holdout["final_1000"]
        and winner_holdout["win_rate"] >= baseline_holdout["win_rate"]
        and winner_holdout["max_drawdown"] >= baseline_holdout["max_drawdown"]
        and neighbor_passes >= 2
    )
    net_log, net_summary = net_account(method_logs[winner])
    _, baseline_net = net_account(baseline)

    fixed.to_csv(OUT / "exit_variants.csv", index=False, encoding="utf-8-sig")
    methods.to_csv(
        OUT / "rolling_methods.csv", index=False, encoding="utf-8-sig"
    )
    method_choices[winner].to_csv(
        OUT / "winner_choices.csv", index=False, encoding="utf-8-sig"
    )
    method_logs[winner].to_csv(
        OUT / "winner_trades.csv", index=False, encoding="utf-8-sig"
    )
    net_log.to_csv(
        OUT / "winner_net_trades.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "control_variable": "main_strategy_exit_only",
        "baseline_gross_final_1000": float(
            1000.0 * np.prod(1.0 + source["ret"].dropna())
        ),
        "baseline_net": baseline_net,
        "baseline_holdout": baseline_holdout,
        "development_selected_method": winner,
        "winner_holdout": winner_holdout,
        "winner_net": net_summary,
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
