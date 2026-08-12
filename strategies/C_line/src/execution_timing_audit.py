from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OUT = ROOT / "fit" / "execution_timing_audit"
INITIAL_CASH = 10_000.0
COMMISSION_RATE = 0.0003
MIN_COMMISSION = 5.0
HOLDOUT = {2024, 2025, 2026}


def read_trades(path: Path) -> pd.DataFrame:
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


def is_main(frame: pd.DataFrame) -> pd.Series:
    if "family" in frame:
        family = frame["family"].astype(str)
        known = family.eq("MAIN")
        if known.any():
            return known
    return frame["source"].astype(str).eq("主策略")


def price_frame(symbol: str, preferred: Path) -> pd.DataFrame:
    path = preferred / "raw" / "etf" / f"{symbol}.csv"
    if not path.exists():
        alternate = WORKSPACE / "S" if preferred == ROOT else ROOT
        path = alternate / "raw" / "etf" / f"{symbol}.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return frame.dropna(subset=["date", "close"]).sort_values("date")


def corrected_timing(
    frame: pd.DataFrame,
    preferred: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    main_mask = is_main(result)
    main_entries = pd.DatetimeIndex(
        result.loc[main_mask, "entry_date"].dropna().sort_values().unique()
    )
    audit_rows = []
    for index, trade in result.loc[~main_mask].iterrows():
        hold = pd.to_numeric(
            pd.Series([trade.get("hold", np.nan)]), errors="coerce"
        ).iloc[0]
        if pd.isna(hold):
            continue
        symbol = str(trade["symbol"]).zfill(6)
        prices = price_frame(symbol, preferred)
        dates = pd.DatetimeIndex(prices["date"])
        entry_pos = int(dates.searchsorted(trade["entry_date"], side="left"))
        if entry_pos >= len(dates):
            continue
        natural_pos = min(entry_pos + int(hold), len(dates) - 1)
        natural_due = pd.Timestamp(dates[natural_pos])
        original_exit = pd.Timestamp(trade["exit_date"])
        clipped = bool(original_exit < natural_due)
        next_main_pos = int(
            main_entries.searchsorted(original_exit, side="right")
        )
        next_main = (
            pd.Timestamp(main_entries[next_main_pos])
            if next_main_pos < len(main_entries)
            else pd.NaT
        )
        corrected_exit = original_exit
        reason = "unchanged"
        if clipped:
            candidates = [natural_due]
            if pd.notna(next_main):
                candidates.append(next_main)
            target = min(candidates)
            price_pos = int(dates.searchsorted(target, side="left"))
            if price_pos < len(dates):
                corrected_exit = pd.Timestamp(dates[price_pos])
                reason = (
                    "same_close_main_switch"
                    if pd.notna(next_main) and target == next_main
                    else "scheduled_max_hold"
                )
        entry_close = float(prices.iloc[entry_pos]["close"])
        exit_pos = int(dates.searchsorted(corrected_exit, side="left"))
        exit_close = float(prices.iloc[exit_pos]["close"])
        original_ret = float(trade["ret"])
        corrected_ret = exit_close / entry_close - 1.0
        result.at[index, "exit_date"] = corrected_exit
        result.at[index, "entry_close"] = entry_close
        result.at[index, "exit_close"] = exit_close
        result.at[index, "ret"] = corrected_ret
        audit_rows.append(
            {
                "symbol": symbol,
                "source": trade.get("source", ""),
                "entry_date": trade["entry_date"],
                "original_exit": original_exit,
                "natural_due": natural_due,
                "next_main_entry": next_main,
                "corrected_exit": corrected_exit,
                "was_clipped": clipped,
                "correction_reason": reason,
                "original_ret": original_ret,
                "corrected_ret": corrected_ret,
                "ret_change": corrected_ret - original_ret,
            }
        )
    return (
        result.sort_values(["entry_date", "symbol"]).reset_index(drop=True),
        pd.DataFrame(audit_rows),
    )


def metrics(frame: pd.DataFrame, years: set[int] | None = None) -> dict:
    local = frame.copy()
    if years is not None:
        local = local[
            pd.to_datetime(local["entry_date"]).dt.year.isin(years)
        ]
    returns = pd.to_numeric(local["ret"], errors="coerce").dropna()
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
        "final_1000": float(equity.iloc[-1]) if len(equity) else 1000.0,
        "win_rate": float((returns > 0).mean()) if len(returns) else np.nan,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "losing_years": int((annual < 0).sum()),
    }


def fee(notional: float) -> float:
    return max(MIN_COMMISSION, abs(notional) * COMMISSION_RATE)


def net_account(frame: pd.DataFrame) -> dict:
    cash = INITIAL_CASH
    executed = 0
    for _, trade in frame.sort_values(["entry_date", "symbol"]).iterrows():
        entry = float(trade.get("entry_close", np.nan))
        exit_ = float(trade.get("exit_close", np.nan))
        if not np.isfinite(entry) or not np.isfinite(exit_):
            prices = price_frame(str(trade["symbol"]).zfill(6), ROOT)
            lookup = prices.set_index("date")["close"]
            entry = float(lookup[pd.Timestamp(trade["entry_date"])])
            exit_ = float(lookup[pd.Timestamp(trade["exit_date"])])
        quantity = math.floor(cash / (entry * 100)) * 100
        while quantity >= 100 and quantity * entry + fee(
            quantity * entry
        ) > cash:
            quantity -= 100
        if quantity < 100:
            continue
        buy_value = quantity * entry
        sell_value = quantity * exit_
        cash += (
            sell_value
            - fee(sell_value)
            - buy_value
            - fee(buy_value)
        )
        executed += 1
    return {
        "initial_cash": INITIAL_CASH,
        "final_cash": float(cash),
        "net_profit": float(cash - INITIAL_CASH),
        "executed_trades": executed,
    }


def summarize(
    name: str,
    original: pd.DataFrame,
    corrected: pd.DataFrame,
    audit: pd.DataFrame,
) -> dict:
    return {
        "line": name,
        "clipped_trades": int(audit["was_clipped"].sum()),
        "original_all": metrics(original),
        "corrected_all": metrics(corrected),
        "original_holdout": metrics(original, HOLDOUT),
        "corrected_holdout": metrics(corrected, HOLDOUT),
        "corrected_net": net_account(corrected),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "C": (
            ROOT / "fit" / "fallback" / "final_trades.csv",
            ROOT,
        ),
        "S": (
            WORKSPACE / "S" / "fit" / "selector" / "final_trades.csv",
            WORKSPACE / "S",
        ),
        "HYBRID": (
            ROOT / "fit" / "hybrid_test" / "winner_trades.csv",
            ROOT,
        ),
    }
    summaries = []
    for name, (path, preferred) in paths.items():
        original = read_trades(path)
        corrected, audit = corrected_timing(original, preferred)
        corrected.to_csv(
            OUT / f"{name.lower()}_corrected_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )
        audit.to_csv(
            OUT / f"{name.lower()}_exit_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
        summaries.append(summarize(name, original, corrected, audit))
    result = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "execution_rule": (
            "fallback exits at the next main entry close when a main signal "
            "preempts its scheduled maximum hold"
        ),
        "lines": summaries,
    }
    (OUT / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
