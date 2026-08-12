from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OUT = ROOT / "fit" / "formal_clean" / "audit.json"


def load_trades(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    exit_col = "actual_exit_date" if "actual_exit_date" in frame else "exit_date"
    frame[exit_col] = pd.to_datetime(frame[exit_col], errors="coerce")
    return frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def quality(path: Path) -> tuple[dict, set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload, {str(value).zfill(6) for value in payload["approved_symbols"]}


def no_overlap(frame: pd.DataFrame) -> bool:
    exit_col = "actual_exit_date" if "actual_exit_date" in frame else "exit_date"
    closed = frame.dropna(subset=[exit_col]).sort_values(["entry_date", exit_col])
    previous = None
    for _, row in closed.iterrows():
        if previous is not None and row["entry_date"] < previous:
            return False
        previous = row[exit_col]
    return True


def gross_return_error(frame: pd.DataFrame) -> float:
    required = {"entry_close", "exit_close", "ret"}
    if not required.issubset(frame.columns):
        return float("nan")
    entry = pd.to_numeric(frame["entry_close"], errors="coerce")
    exit_ = pd.to_numeric(frame["exit_close"], errors="coerce")
    stated = pd.to_numeric(frame["ret"], errors="coerce")
    mask = entry.gt(0) & exit_.notna() & stated.notna()
    return float(((exit_[mask] / entry[mask] - 1.0) - stated[mask]).abs().max())


def d_pnl_error(frame: pd.DataFrame) -> float:
    closed = frame[frame["status_portfolio"].eq("closed")].copy()
    quantity = pd.to_numeric(closed["quantity"], errors="coerce")
    entry = pd.to_numeric(closed["entry_close"], errors="coerce")
    exit_ = pd.to_numeric(closed["actual_exit_close"], errors="coerce")
    buy_fee = pd.to_numeric(closed["buy_fee"], errors="coerce")
    sell_fee = pd.to_numeric(closed["sell_fee"], errors="coerce")
    stated = pd.to_numeric(closed["net_pnl"], errors="coerce")
    computed = quantity * (exit_ - entry) - buy_fee - sell_fee
    return float((computed - stated).abs().max())


def raw_price_error(frame: pd.DataFrame, preferred: Path) -> float:
    cache: dict[str, pd.Series] = {}
    errors: list[float] = []
    exit_date_col = "actual_exit_date" if "actual_exit_date" in frame else "exit_date"
    exit_price_col = "actual_exit_close" if "actual_exit_close" in frame else "exit_close"
    for _, row in frame.iterrows():
        symbol = str(row["symbol"]).zfill(6)
        if symbol not in cache:
            path = preferred / "raw" / "etf" / f"{symbol}.csv"
            if not path.exists():
                alternate = WORKSPACE / "S" if preferred == ROOT else ROOT
                path = alternate / "raw" / "etf" / f"{symbol}.csv"
            prices = pd.read_csv(path, encoding="utf-8-sig")
            prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
            prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
            cache[symbol] = prices.dropna(subset=["date", "close"]).set_index("date")["close"]
        lookup = cache[symbol]
        entry = pd.to_numeric(pd.Series([row.get("entry_close")]), errors="coerce").iloc[0]
        if pd.notna(entry):
            errors.append(abs(float(entry) - float(lookup[pd.Timestamp(row["entry_date"])])))
        exit_date = row.get(exit_date_col)
        exit_price = pd.to_numeric(pd.Series([row.get(exit_price_col)]), errors="coerce").iloc[0]
        if pd.notna(exit_date) and pd.notna(exit_price):
            errors.append(abs(float(exit_price) - float(lookup[pd.Timestamp(exit_date)])))
    return max(errors, default=0.0)


def past_only(path: Path) -> bool:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    year = pd.to_numeric(frame["year"], errors="coerce")
    train_end = pd.to_numeric(frame["train_end"], errors="coerce")
    return bool((train_end < year).all())


def main() -> None:
    c_quality, c_approved = quality(ROOT / "raw" / "quality.json")
    s_quality, s_approved = quality(WORKSPACE / "S" / "raw" / "quality.json")
    c = load_trades(ROOT / "fit" / "formal_clean" / "final_trades.csv")
    s = load_trades(WORKSPACE / "S" / "fit" / "selector" / "formal_execution_trades.csv")
    d = load_trades(WORKSPACE / "D" / "out" / "formal_clean" / "account_trades.csv")

    checks = {
        "c_quality_gate": bool(c_quality["passed"]),
        "s_quality_gate": bool(s_quality["passed"]),
        "c_symbols_approved": set(c["symbol"]).issubset(c_approved),
        "s_symbols_approved": set(s["symbol"]).issubset(s_approved),
        "d_symbols_approved": set(d["symbol"]).issubset(c_approved),
        "excluded_512690_absent": "512690" not in set(pd.concat([c["symbol"], s["symbol"], d["symbol"]])),
        "c_chronology": bool((c["entry_date"] <= c["exit_date"]).all()),
        "s_chronology": bool((s["entry_date"] <= s["exit_date"]).all()),
        "d_chronology": bool(
            (d.loc[d["actual_exit_date"].notna(), "entry_date"] <= d.loc[d["actual_exit_date"].notna(), "actual_exit_date"]).all()
        ),
        "c_single_position": no_overlap(c),
        "s_single_position": no_overlap(s),
        "d_single_position": no_overlap(d),
        "s_selector_past_only": past_only(WORKSPACE / "S" / "fit" / "selector" / "selected_by_year.csv"),
        "d_selector_past_only": past_only(WORKSPACE / "D" / "out" / "formal_clean" / "selected_by_year.csv"),
        "c_return_replay": gross_return_error(c) < 1e-10,
        "s_return_replay": gross_return_error(s) < 1e-10,
        "d_net_pnl_replay": d_pnl_error(d) < 1e-8,
        "c_raw_price_replay": raw_price_error(c, ROOT) < 1e-10,
        "s_raw_price_replay": raw_price_error(s, WORKSPACE / "S") < 1e-10,
        "d_raw_price_replay": raw_price_error(d, ROOT) < 1e-10,
    }
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "passed": all(checks.values()),
        "checks": checks,
        "maximum_errors": {
            "c_return": gross_return_error(c),
            "s_return": gross_return_error(s),
            "d_net_pnl": d_pnl_error(d),
            "c_raw_price": raw_price_error(c, ROOT),
            "s_raw_price": raw_price_error(s, WORKSPACE / "S"),
            "d_raw_price": raw_price_error(d, ROOT),
        },
        "trade_rows": {"C": len(c), "S": len(s), "D": len(d)},
        "note": "C main/fallback annual rolling selection contract is frozen in the original selector audit; this audit rechecks its clean-price execution table.",
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
