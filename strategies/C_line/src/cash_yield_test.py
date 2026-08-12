from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import take_profit_10_test as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "cash_yield"


def simulate(frame: pd.DataFrame, annual_yield: float) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cash = core.INITIAL_CAPITAL
    previous_exit: pd.Timestamp | None = None
    rows = []
    for _, trade in frame.sort_values(["entry_date", "symbol"]).iterrows():
        entry_date = pd.Timestamp(trade["entry_date"])
        cycle_start = cash
        idle_days = max((entry_date - previous_exit).days, 0) if previous_exit is not None else 0
        idle_interest = cash * ((1.0 + annual_yield) ** (idle_days / 365.0) - 1.0)
        cash += idle_interest
        entry = float(trade["entry_close"])
        exit_price = float(trade["exit_close_test"])
        quantity = math.floor(cash / (entry * 100)) * 100
        while quantity >= 100 and quantity * entry + core.commission(quantity * entry) > cash:
            quantity -= 100
        if quantity < 100:
            continue
        buy_value = quantity * entry
        buy_fee = core.commission(buy_value)
        cash -= buy_value + buy_fee
        mark_value = quantity * exit_price
        if bool(trade["open_mark_test"]):
            after = cash + mark_value - core.commission(mark_value)
        else:
            cash += mark_value - core.commission(mark_value)
            after = cash
        item = trade.to_dict()
        item.update(
            {
                "cash_yield_rate": annual_yield,
                "idle_calendar_days": idle_days,
                "idle_interest": idle_interest,
                "quantity_cash_test": quantity,
                "cash_after_cash_test": after,
                "cycle_return_cash_test": after / cycle_start - 1.0,
            }
        )
        rows.append(item)
        previous_exit = pd.Timestamp(trade["exit_date_test"])
    detail = pd.DataFrame(rows)
    detail["year"] = pd.to_datetime(detail["entry_date"]).dt.year
    annual = (
        detail.groupby("year")["cycle_return_cash_test"]
        .apply(lambda values: float(np.prod(1.0 + values.astype(float)) - 1.0))
        .reindex(range(int(detail["year"].min()), 2027), fill_value=0.0)
        .rename("account_return")
        .reset_index()
    )
    equity = detail["cash_after_cash_test"].astype(float)
    closed = detail[~detail["open_mark_test"].astype(bool)]
    gross = pd.to_numeric(closed["gross_return_test"], errors="coerce")
    stats = {
        "trades": len(detail),
        "final_value": float(detail.iloc[-1]["cash_after_cash_test"]),
        "win_rate": float((gross > 0).mean()),
        "avg_annual_return": float(annual["account_return"].mean()),
        "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
        "idle_days": int(detail["idle_calendar_days"].sum()),
        "idle_interest_total": float(detail["idle_interest"].sum()),
    }
    return stats, detail, annual


def main() -> None:
    core.configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in ("C", "S", "D"):
        path = ROOT / "fit" / "rolling_upgrades" / f"{line.lower()}_trades.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"symbol": str})
        frame["symbol"] = frame["symbol"].str.zfill(6)
        for column in ("entry_date", "exit_date_test"):
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        for rate in (0.0, 0.005, 0.01, 0.015, 0.02):
            stats, detail, annual = simulate(frame, rate)
            rows.append({"line": line, "annual_yield": rate, **stats})
            if rate == 0.01:
                detail.to_csv(OUT / f"{line.lower()}_trades_1pct.csv", index=False, encoding="utf-8-sig")
                annual.to_csv(OUT / f"{line.lower()}_annual_1pct.csv", index=False, encoding="utf-8-sig")
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "sensitivity.csv", index=False, encoding="utf-8-sig")
    payload = {
        "method": "idle all-cash periods accrue a net annual cash-management yield; strategy trades unchanged",
        "boundary": "first pre-strategy idle period and residual cash while invested receive no yield",
        "results": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
