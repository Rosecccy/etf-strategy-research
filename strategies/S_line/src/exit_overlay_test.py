from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
BASE = WORKSPACE / "C" / "fit" / "execution_timing_audit" / "s_corrected_trades.csv"
OUT = ROOT / "fit" / "exit_overlay_test"
DEV = set(range(2016, 2024))
HOLDOUT = {2024, 2025, 2026}
INITIAL = 10_000.0
FEE_RATE = 0.0003
MIN_FEE = 5.0


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def load_trades() -> pd.DataFrame:
    frame = pd.read_csv(BASE, dtype={"symbol": str}, encoding="utf-8-sig")
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    frame["exit_date"] = pd.to_datetime(frame["exit_date"], errors="coerce")
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    return frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


@lru_cache(maxsize=None)
def price_frame(symbol: str) -> pd.DataFrame:
    candidates = [
        ROOT / "raw" / "etf" / f"{symbol}.csv",
        WORKSPACE / "C" / "raw" / "etf" / f"{symbol}.csv",
    ]
    path = next((item for item in candidates if item.exists()), None)
    if path is None:
        raise FileNotFoundError(symbol)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date")
    close = frame["close"]
    frame["ma20"] = close.rolling(20, min_periods=15).mean()
    frame["ma40"] = close.rolling(40, min_periods=25).mean()
    frame["ma60"] = close.rolling(60, min_periods=40).mean()
    frame["ret20"] = close.pct_change(20)
    frame["ret40"] = close.pct_change(40)
    return frame.reset_index(drop=True)


def next_main_dates(frame: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        frame.loc[frame["source"].astype(str).eq("主策略"), "entry_date"]
        .dropna()
        .sort_values()
        .unique()
    )


def next_main_after(
    entries: pd.DatetimeIndex,
    entry_date: pd.Timestamp,
) -> pd.Timestamp | None:
    pos = int(entries.searchsorted(entry_date, side="right"))
    return pd.Timestamp(entries[pos]) if pos < len(entries) else None


def overlay_trade(
    trade: pd.Series,
    main_entries: pd.DatetimeIndex,
    mode: str,
    min_hold: int,
    threshold: float,
    extension: int,
    ma_days: int,
) -> dict:
    item = trade.to_dict()
    if str(trade["source"]) == "主策略":
        item["overlay_reason"] = "main_unchanged"
        return item
    prices = price_frame(str(trade["symbol"]))
    dates = pd.DatetimeIndex(prices["date"])
    entry_pos = int(dates.searchsorted(trade["entry_date"], side="left"))
    original_exit_pos = int(dates.searchsorted(trade["exit_date"], side="left"))
    if entry_pos >= len(prices) or original_exit_pos >= len(prices):
        item["overlay_reason"] = "missing_price"
        return item
    entry_price = float(prices.iloc[entry_pos]["close"])
    new_exit_pos = original_exit_pos
    reason = "original_exit"

    if mode == "early_momentum":
        last_signal = max(entry_pos + min_hold, entry_pos)
        for signal_pos in range(last_signal, original_exit_pos):
            row = prices.iloc[signal_pos]
            ma_value = row[f"ma{ma_days}"]
            if (
                pd.notna(row["ret20"])
                and pd.notna(ma_value)
                and float(row["ret20"]) <= threshold
                and float(row["close"]) < float(ma_value)
            ):
                new_exit_pos = min(signal_pos + 1, original_exit_pos)
                reason = "early_momentum_t1"
                break
    elif mode == "early_trailing":
        peak = entry_price
        last_signal = max(entry_pos + min_hold, entry_pos)
        for signal_pos in range(entry_pos, original_exit_pos):
            close = float(prices.iloc[signal_pos]["close"])
            peak = max(peak, close)
            if signal_pos >= last_signal and close / peak - 1.0 <= -threshold:
                new_exit_pos = min(signal_pos + 1, original_exit_pos)
                reason = "early_trailing_t1"
                break
    elif mode == "strong_extend":
        signal = prices.iloc[original_exit_pos]
        ma_value = signal[f"ma{ma_days}"]
        if (
            pd.notna(signal["ret20"])
            and pd.notna(ma_value)
            and float(signal["ret20"]) >= threshold
            and float(signal["close"]) > float(ma_value)
        ):
            cap_pos = min(original_exit_pos + extension, len(prices) - 1)
            next_main = next_main_after(main_entries, trade["entry_date"])
            if next_main is not None:
                cap_pos = min(
                    cap_pos,
                    int(dates.searchsorted(next_main, side="left")),
                )
            new_exit_pos = max(original_exit_pos, cap_pos)
            reason = "strong_extension"

    exit_price = float(prices.iloc[new_exit_pos]["close"])
    item["exit_date"] = pd.Timestamp(prices.iloc[new_exit_pos]["date"])
    item["entry_close"] = entry_price
    item["exit_close"] = exit_price
    item["ret"] = exit_price / entry_price - 1.0
    item["overlay_reason"] = reason
    return item


def apply_overlay(
    frame: pd.DataFrame,
    mode: str,
    min_hold: int,
    threshold: float,
    extension: int,
    ma_days: int,
) -> pd.DataFrame:
    main_entries = next_main_dates(frame)
    rows = [
        overlay_trade(
            row,
            main_entries,
            mode,
            min_hold,
            threshold,
            extension,
            ma_days,
        )
        for _, row in frame.iterrows()
    ]
    result = pd.DataFrame(rows).sort_values(["entry_date", "symbol"])
    kept: list[pd.Series] = []
    busy = pd.Timestamp.min
    for _, row in result.iterrows():
        if pd.Timestamp(row["entry_date"]) < busy:
            continue
        kept.append(row)
        busy = pd.Timestamp(row["exit_date"])
    return pd.DataFrame(kept).reset_index(drop=True)


def metrics(frame: pd.DataFrame, years: set[int]) -> dict:
    local = frame[
        pd.to_datetime(frame["entry_date"]).dt.year.isin(years)
    ].copy()
    returns = pd.to_numeric(local["ret"], errors="coerce").dropna()
    equity = 1000.0 * (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    annual = (
        local.assign(year=local["entry_date"].dt.year)
        .groupby("year")["ret"]
        .apply(lambda values: float(np.prod(1.0 + values) - 1.0))
    )
    return {
        "trades": int(len(returns)),
        "final_1000": float(equity.iloc[-1]) if len(equity) else 1000.0,
        "win_rate": float((returns > 0).mean()) if len(returns) else np.nan,
        "avg_return": float(returns.mean()) if len(returns) else np.nan,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "losing_years": int((annual < 0).sum()),
    }


def score(stat: dict) -> float:
    return (
        math.log(max(float(stat["final_1000"]), 1.0) / 1000.0)
        + 0.30 * float(np.nan_to_num(stat["win_rate"], nan=0.0))
        + 0.70 * float(stat["max_drawdown"])
        - 0.05 * int(stat["losing_years"])
    )


def fee(value: float) -> float:
    return max(MIN_FEE, abs(value) * FEE_RATE)


def net_account(frame: pd.DataFrame) -> dict:
    cash = INITIAL
    executed = 0
    for _, row in frame.sort_values(["entry_date", "symbol"]).iterrows():
        entry = float(row.get("entry_close", np.nan))
        exit_ = float(row.get("exit_close", np.nan))
        if not np.isfinite(entry) or not np.isfinite(exit_):
            prices = price_frame(str(row["symbol"]))
            lookup = prices.set_index("date")["close"]
            entry = float(lookup[pd.Timestamp(row["entry_date"])])
            exit_ = float(lookup[pd.Timestamp(row["exit_date"])])
        qty = math.floor(cash / (entry * 100)) * 100
        while qty >= 100 and qty * entry + fee(qty * entry) > cash:
            qty -= 100
        if qty < 100:
            continue
        cash += qty * exit_ - fee(qty * exit_) - qty * entry - fee(qty * entry)
        executed += 1
    return {
        "initial_cash": INITIAL,
        "final_cash": float(cash),
        "net_profit": float(cash - INITIAL),
        "executed_trades": executed,
    }


def main() -> None:
    configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = load_trades()
    variants: list[tuple[str, int, float, int, int]] = [
        ("baseline", 0, 0.0, 0, 20)
    ]
    for minimum in (3, 5, 8, 10, 15, 20):
        for cutoff in (-0.08, -0.05, -0.03, 0.0, 0.03):
            for ma in (20, 40, 60):
                variants.append(("early_momentum", minimum, cutoff, 0, ma))
        for drawdown in (0.03, 0.05, 0.07, 0.10, 0.12):
            variants.append(("early_trailing", minimum, drawdown, 0, 20))
    for cutoff in (0.0, 0.03, 0.05, 0.08, 0.10, 0.15):
        for days in (3, 5, 8, 10, 15, 20, 30):
            for ma in (20, 40, 60):
                variants.append(("strong_extend", 0, cutoff, days, ma))

    rows = []
    accounts: dict[str, pd.DataFrame] = {}
    for mode, minimum, threshold, extension, ma in variants:
        variant_id = (
            f"{mode}_min{minimum}_t{threshold:+.2f}_"
            f"x{extension}_ma{ma}"
        )
        account = (
            baseline.copy()
            if mode == "baseline"
            else apply_overlay(
                baseline, mode, minimum, threshold, extension, ma
            )
        )
        dev = metrics(account, DEV)
        holdout = metrics(account, HOLDOUT)
        all_stat = metrics(account, DEV | HOLDOUT)
        rows.append(
            {
                "variant_id": variant_id,
                "mode": mode,
                "min_hold": minimum,
                "threshold": threshold,
                "extension": extension,
                "ma_days": ma,
                "dev_score": score(dev),
                **{f"dev_{key}": value for key, value in dev.items()},
                **{
                    f"holdout_{key}": value
                    for key, value in holdout.items()
                },
                **{f"all_{key}": value for key, value in all_stat.items()},
            }
        )
        accounts[variant_id] = account

    table = pd.DataFrame(rows).sort_values(
        ["dev_score", "dev_final_1000", "dev_win_rate"],
        ascending=False,
    )
    winner = table.iloc[0]
    winner_id = str(winner["variant_id"])
    winner_account = accounts[winner_id]
    base_row = table[table["mode"].eq("baseline")].iloc[0]
    nearby = table[
        table["mode"].eq(winner["mode"])
        & table["dev_final_1000"].ge(float(winner["dev_final_1000"]) * 0.95)
    ]
    passing_neighbors = nearby[
        nearby["holdout_final_1000"].gt(base_row["holdout_final_1000"])
        & nearby["holdout_win_rate"].ge(base_row["holdout_win_rate"])
        & nearby["holdout_max_drawdown"].ge(base_row["holdout_max_drawdown"])
    ]
    promoted = bool(
        winner["holdout_final_1000"] > base_row["holdout_final_1000"] * 1.03
        and winner["holdout_win_rate"] >= base_row["holdout_win_rate"]
        and winner["holdout_max_drawdown"] >= base_row["holdout_max_drawdown"]
        and len(passing_neighbors) >= 3
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test": "S fixed-entry single-variable exit overlay",
        "candidate_variants": int(len(table)),
        "development_years": "2016-2023",
        "holdout_years": "2024-2026",
        "baseline": base_row.to_dict(),
        "development_winner": winner.to_dict(),
        "nearby_development_variants": int(len(nearby)),
        "passing_neighbors": int(len(passing_neighbors)),
        "winner_net_10000": net_account(winner_account),
        "baseline_net_10000": net_account(baseline),
        "promoted": promoted,
    }
    table.to_csv(OUT / "variants.csv", index=False, encoding="utf-8-sig")
    winner_account.to_csv(
        OUT / "winner_trades.csv", index=False, encoding="utf-8-sig"
    )
    passing_neighbors.to_csv(
        OUT / "passing_neighbors.csv", index=False, encoding="utf-8-sig"
    )
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
