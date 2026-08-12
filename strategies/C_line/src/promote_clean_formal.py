from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
TIMING = ROOT / "fit" / "execution_timing_audit"
C_FORMAL = ROOT / "fit" / "formal_clean"
S_FORMAL = WORKSPACE / "S" / "fit" / "selector"
HOLDOUT_YEARS = {2024, 2025, 2026}
INITIAL_CASH = 10_000.0
COMMISSION_RATE = 0.0003
MIN_COMMISSION = 5.0


def read_trades(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    frame["exit_date"] = pd.to_datetime(frame["exit_date"], errors="coerce")
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    return frame.dropna(subset=["entry_date", "exit_date", "ret"]).sort_values(
        ["entry_date", "exit_date", "symbol"]
    ).reset_index(drop=True)


def reprice(frame: pd.DataFrame, preferred: Path) -> pd.DataFrame:
    result = frame.copy()
    cache: dict[str, pd.Series] = {}
    for index, trade in result.iterrows():
        symbol = str(trade["symbol"]).zfill(6)
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
        entry = float(lookup[pd.Timestamp(trade["entry_date"])])
        exit_ = float(lookup[pd.Timestamp(trade["exit_date"])])
        result.at[index, "entry_close"] = entry
        result.at[index, "exit_close"] = exit_
        result.at[index, "ret"] = exit_ / entry - 1.0
    result["cash_after"] = 1000.0 * (1.0 + result["ret"].astype(float)).cumprod()
    return result


def fee(notional: float) -> float:
    return max(MIN_COMMISSION, abs(notional) * COMMISSION_RATE)


def net_account(frame: pd.DataFrame) -> dict:
    cash = INITIAL_CASH
    executed = 0
    for _, trade in frame.sort_values(["entry_date", "symbol"]).iterrows():
        entry = float(trade["entry_close"])
        exit_ = float(trade["exit_close"])
        quantity = math.floor(cash / (entry * 100)) * 100
        while quantity >= 100 and quantity * entry + fee(quantity * entry) > cash:
            quantity -= 100
        if quantity < 100:
            continue
        buy_value = quantity * entry
        sell_value = quantity * exit_
        cash += sell_value - fee(sell_value) - buy_value - fee(buy_value)
        executed += 1
    return {
        "initial_cash": INITIAL_CASH,
        "final_cash": float(cash),
        "net_profit": float(cash - INITIAL_CASH),
        "executed_trades": executed,
    }


def metrics(frame: pd.DataFrame, years: set[int] | None = None) -> dict:
    local = frame.copy()
    if years is not None:
        local = local[local["entry_date"].dt.year.isin(years)]
    values = local["ret"].astype(float)
    if values.empty:
        return {"trades": 0, "final_1000": 1000.0, "win_rate": None, "max_drawdown": 0.0}
    equity = 1000.0 * (1.0 + values).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    annual = local.groupby(local["entry_date"].dt.year)["ret"].apply(
        lambda series: float(np.prod(1.0 + series.astype(float)) - 1.0)
    )
    return {
        "trades": int(len(values)),
        "final_1000": float(equity.iloc[-1]),
        "win_rate": float((values > 0).mean()),
        "avg_return": float(values.mean()),
        "max_drawdown": float(drawdown.min()),
        "losing_years": int((annual < 0).sum()),
    }


def annual_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in frame.groupby(frame["entry_date"].dt.year, sort=True):
        values = group["ret"].astype(float)
        rows.append(
            {
                "year": int(year),
                "trades": int(len(values)),
                "win_rate": float((values > 0).mean()),
                "gross_return": float(np.prod(1.0 + values) - 1.0),
            }
        )
    return pd.DataFrame(rows)


def assert_clean(frame: pd.DataFrame, manifest: Path, label: str) -> dict:
    quality = json.loads(manifest.read_text(encoding="utf-8"))
    if not bool(quality.get("passed")):
        raise RuntimeError(f"{label} quality gate failed.")
    approved = {str(value).zfill(6) for value in quality["approved_symbols"]}
    unknown = sorted(set(frame["symbol"]) - approved)
    if unknown:
        raise RuntimeError(f"{label} formal history contains unapproved ETFs: {unknown}")
    return quality


def main() -> None:
    timing = json.loads((TIMING / "summary.json").read_text(encoding="utf-8"))
    timing_map = {item["line"]: item for item in timing["lines"]}
    c = reprice(read_trades(TIMING / "c_corrected_trades.csv"), ROOT)
    s = reprice(read_trades(TIMING / "s_corrected_trades.csv"), WORKSPACE / "S")
    c_quality = assert_clean(c, ROOT / "raw" / "quality.json", "C")
    s_quality = assert_clean(s, WORKSPACE / "S" / "raw" / "quality.json", "S")

    C_FORMAL.mkdir(parents=True, exist_ok=True)
    c.to_csv(C_FORMAL / "final_trades.csv", index=False, encoding="utf-8-sig")
    annual_table(c).to_csv(C_FORMAL / "annual.csv", index=False, encoding="utf-8-sig")
    c_summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "formal_clean",
        "version": "C2.4-clean-20260801",
        "data_quality_passed": True,
        "approved_universe": len(c_quality["approved_symbols"]),
        "selection": "Frozen strict rolling C main plus C fallback; clean-price replay and execution correction.",
        "development_years": "2014-2023",
        "holdout_years": "2024-2026",
        "all": metrics(c),
        "holdout": metrics(c, HOLDOUT_YEARS),
        "net_10000": net_account(c),
        "challenger": json.loads((ROOT / "fit" / "strengthen" / "summary.json").read_text(encoding="utf-8"))["winner"],
        "governance": "The clean challenger was rejected because it reduced holdout value, win rate and drawdown stability.",
    }
    (C_FORMAL / "summary.json").write_text(
        json.dumps(c_summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    s.to_csv(S_FORMAL / "formal_execution_trades.csv", index=False, encoding="utf-8-sig")
    annual_table(s).to_csv(S_FORMAL / "annual.csv", index=False, encoding="utf-8-sig")
    existing_s = json.loads((S_FORMAL / "summary.json").read_text(encoding="utf-8"))
    existing_s["execution_corrected"] = {
        "rule": "A fallback position exits at an incoming main entry close before the main purchase.",
        "all": metrics(s),
        "holdout": metrics(s, HOLDOUT_YEARS),
        "net_10000": net_account(s),
    }
    (S_FORMAL / "summary.json").write_text(
        json.dumps(existing_s, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    s_cfg_path = WORKSPACE / "S" / "cfg" / "s1_model.json"
    s_cfg = json.loads(s_cfg_path.read_text(encoding="utf-8-sig"))
    strict = s_cfg["strict_validation"]
    strict.update(
        {
            "gross_final_cash_1000": metrics(s)["final_1000"],
            "net_final_cash_10000": net_account(s)["final_cash"],
            "trade_count": metrics(s)["trades"],
            "win_rate": metrics(s)["win_rate"],
            "max_drawdown": metrics(s)["max_drawdown"],
            "holdout_return": metrics(s, HOLDOUT_YEARS)["final_1000"] / 1000.0 - 1.0,
            "source": "fit/selector/summary.json#execution_corrected",
        }
    )
    s_cfg_path.write_text(json.dumps(s_cfg, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    active_path = ROOT / "cfg" / "active.json"
    active = {
        "formal_account_strategy": "C2.4-clean-20260801",
        "formal_config": "C/cfg/live_model.json",
        "daily_decision": "C/live/today_decision.csv",
        "daily_runner": "C/src/run_all_daily.py",
        "component_engines": {"C2.4": "clean strict rolling main and C fallback"},
        "rule": "C/S hybrid clean retrain failed the promotion gate; C2.4 is the formal C instruction.",
        "activated_on": "2026-08-01",
    }
    active_path.write_text(json.dumps(active, ensure_ascii=False, indent=2), encoding="utf-8")

    hybrid_cfg_path = ROOT / "cfg" / "hybrid.json"
    hybrid_cfg = json.loads(hybrid_cfg_path.read_text(encoding="utf-8-sig"))
    hybrid_cfg["status"] = "research_not_promoted_after_clean_retrain"
    hybrid_cfg["validation"] = json.loads(
        (ROOT / "fit" / "hybrid_formal" / "summary.json").read_text(encoding="utf-8")
    )
    hybrid_cfg_path.write_text(json.dumps(hybrid_cfg, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"C": c_summary, "S": existing_s}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
