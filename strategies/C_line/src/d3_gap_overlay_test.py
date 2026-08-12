from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
D_SRC = WORKSPACE / "D" / "src"
OUT = ROOT / "fit" / "d3_gap_overlay_test"
BASELINE_PATH = ROOT / "fit" / "hybrid_formal" / "final_trades.csv"
INITIAL_CASH = 10_000.0
COMMISSION_RATE = 0.0003
MIN_COMMISSION = 5.0
HOLDOUT_YEARS = {2024, 2025, 2026}

if str(D_SRC) not in sys.path:
    sys.path.insert(0, str(D_SRC))

import daily_panic_rolling_test as d_base  # noqa: E402
import daily_panic_stale_exit_test as d_stale  # noqa: E402


def read_baseline() -> pd.DataFrame:
    frame = pd.read_csv(
        BASELINE_PATH, dtype={"symbol": str}, encoding="utf-8-sig"
    )
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["entry_date"] = pd.to_datetime(frame["entry_date"])
    frame["exit_date"] = pd.to_datetime(frame["exit_date"])
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    frame["entry_close"] = pd.to_numeric(
        frame["entry_close"], errors="coerce"
    )
    frame["exit_close"] = pd.to_numeric(
        frame["exit_close"], errors="coerce"
    )
    return frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def build_d3_history() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cached_closed = OUT / "d3_rolling_closed.csv"
    cached_open = OUT / "d3_current_open.csv"
    cached_choices = OUT / "d3_choices.csv"
    if cached_closed.exists() and cached_open.exists() and cached_choices.exists():
        closed = pd.read_csv(cached_closed, dtype={"symbol": str}, encoding="utf-8-sig")
        open_marked = pd.read_csv(cached_open, dtype={"symbol": str}, encoding="utf-8-sig")
        for frame in (closed, open_marked):
            frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
            frame["entry_date"] = pd.to_datetime(frame["entry_date"])
            frame["exit_date"] = pd.to_datetime(frame["exit_date"])
        return closed, open_marked, pd.read_csv(cached_choices, encoding="utf-8-sig")
    adjusted, raw, pool = d_base.prepare()
    formal = d_stale.build_formal(adjusted, raw, pool)
    logs = {
        rule.key: d_stale.adjust_exits(formal, raw, rule)
        for rule in d_stale.RULES
    }
    selected, choices = d_stale.rolling(formal, logs, raw, pool, None)
    portfolio, _ = d_stale.portfolio_metrics(selected, raw, pool)
    result = portfolio.copy()
    result["symbol"] = result["symbol"].astype(str).str.zfill(6)
    result["entry_date"] = pd.to_datetime(result["entry_date"])
    result["exit_date"] = pd.to_datetime(result["actual_exit_date"])
    result["entry_close"] = pd.to_numeric(
        result["entry_close"], errors="coerce"
    )
    result["exit_close"] = pd.to_numeric(
        result["actual_exit_close"], errors="coerce"
    )
    result["ret"] = result["exit_close"] / result["entry_close"] - 1.0
    result["status"] = result["status_portfolio"].astype(str)
    result["source"] = "空仓补偿_D3"
    result["family"] = "D"
    result["priority"] = 900.0
    result["display_name"] = result.get("name", result["symbol"])
    closed = result[result["status"].eq("closed")].copy()
    open_marked = result[result["status"].eq("open_marked")].copy()
    keep = [
        "source",
        "family",
        "symbol",
        "display_name",
        "entry_date",
        "exit_date",
        "entry_close",
        "exit_close",
        "ret",
        "priority",
        "status",
    ]
    return (
        closed[keep].sort_values("entry_date").reset_index(drop=True),
        open_marked[keep].sort_values("entry_date").reset_index(drop=True),
        choices,
    )


def price_on_or_after(symbol: str, date: pd.Timestamp) -> tuple[pd.Timestamp, float]:
    path = (
        WORKSPACE
        / "D"
        / "out"
        / "research_30_extrema_cleaned_v2"
        / "raw"
        / f"{symbol}.csv"
    )
    if not path.exists():
        path = WORKSPACE / "D" / "raw" / "etf" / f"{symbol}.csv"
    if not path.exists():
        path = ROOT / "raw" / "etf" / f"{symbol}.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date")
    pos = int(pd.DatetimeIndex(frame["date"]).searchsorted(date, side="left"))
    if pos >= len(frame):
        raise ValueError(f"{symbol} has no price on or after {date.date()}")
    row = frame.iloc[pos]
    return pd.Timestamp(row["date"]), float(row["close"])


def historical_gate(
    history: pd.DataFrame, date: pd.Timestamp, mode: str
) -> tuple[bool, float]:
    past = history[history["exit_date"].lt(date)].sort_values("exit_date")
    if mode == "none":
        return True, 0.0
    lookback = {"positive_3": 3, "positive_5": 5, "positive_all": None}[mode]
    if lookback is not None:
        past = past.tail(lookback)
    returns = pd.to_numeric(past["ret"], errors="coerce").dropna()
    if len(returns) < 3:
        return False, -np.inf
    mean = float(returns.mean())
    win = float((returns > 0).mean())
    downside = float(np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)))
    score = mean + 0.10 * (win - 0.5) - 0.25 * downside
    return bool(mean > 0 and score > 0), score


def insert_d3_gaps(
    baseline: pd.DataFrame, d3: pd.DataFrame, mode: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = baseline.sort_values("entry_date").reset_index(drop=True)
    selected: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    d_busy_until = pd.Timestamp.min
    for _, trade in d3.sort_values("entry_date").iterrows():
        entry = pd.Timestamp(trade["entry_date"])
        if entry < d_busy_until:
            audit.append(
                {"entry_date": entry, "symbol": trade["symbol"], "accepted": False, "reason": "D3_busy"}
            )
            continue
        occupied = base[
            base["entry_date"].le(entry) & base["exit_date"].gt(entry)
        ]
        same_day_base = base[base["entry_date"].eq(entry)]
        if not occupied.empty or not same_day_base.empty:
            audit.append(
                {"entry_date": entry, "symbol": trade["symbol"], "accepted": False, "reason": "baseline_busy"}
            )
            continue
        allowed, score = historical_gate(d3, entry, mode)
        if not allowed:
            audit.append(
                {"entry_date": entry, "symbol": trade["symbol"], "accepted": False, "reason": "history_gate", "history_score": score}
            )
            continue
        later = base[base["entry_date"].gt(entry)]
        next_base = (
            pd.Timestamp(later.iloc[0]["entry_date"])
            if not later.empty
            else pd.NaT
        )
        original_exit = pd.Timestamp(trade["exit_date"])
        clipped = pd.notna(next_base) and next_base < original_exit
        item = trade.to_dict()
        if clipped:
            exit_date, exit_close = price_on_or_after(
                str(trade["symbol"]), next_base
            )
            item["exit_date"] = exit_date
            item["exit_close"] = exit_close
            item["ret"] = exit_close / float(trade["entry_close"]) - 1.0
            item["exit_reason"] = "baseline_preemption"
        else:
            item["exit_reason"] = "D3_exit"
        item["history_gate"] = mode
        item["history_score"] = score
        selected.append(item)
        d_busy_until = pd.Timestamp(item["exit_date"])
        audit.append(
            {
                "entry_date": entry,
                "symbol": trade["symbol"],
                "accepted": True,
                "reason": item["exit_reason"],
                "history_score": score,
                "exit_date": item["exit_date"],
                "ret": item["ret"],
            }
        )
    combined = pd.concat(
        [base, pd.DataFrame(selected)], ignore_index=True, sort=False
    ).sort_values(["entry_date", "symbol"]).reset_index(drop=True)
    return combined, pd.DataFrame(audit)


def annual_returns(frame: pd.DataFrame) -> pd.Series:
    start = int(frame["entry_date"].dt.year.min())
    end = int(frame["entry_date"].dt.year.max())
    annual = (
        frame.assign(year=frame["entry_date"].dt.year)
        .groupby("year")["ret"]
        .apply(lambda values: float(np.prod(1.0 + values.astype(float)) - 1.0))
    )
    return annual.reindex(range(start, end + 1), fill_value=0.0)


def metrics(frame: pd.DataFrame, years: set[int] | None = None) -> dict[str, float | int]:
    local = frame.copy()
    if years is not None:
        local = local[local["entry_date"].dt.year.isin(years)]
    returns = pd.to_numeric(local["ret"], errors="coerce").dropna()
    equity = 1000.0 * (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    annual = annual_returns(local) if len(local) else pd.Series(dtype=float)
    return {
        "trades": int(len(returns)),
        "final_1000": float(equity.iloc[-1]) if len(equity) else 1000.0,
        "win_rate": float((returns > 0).mean()) if len(returns) else np.nan,
        "avg_trade_return": float(returns.mean()) if len(returns) else np.nan,
        "avg_annual_return": float(annual.mean()) if len(annual) else np.nan,
        "cagr": float((float(equity.iloc[-1]) / 1000.0) ** (1.0 / len(annual)) - 1.0) if len(equity) and len(annual) else np.nan,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "losing_years": int((annual < 0).sum()),
    }


def fee(value: float) -> float:
    return max(MIN_COMMISSION, abs(value) * COMMISSION_RATE)


def net_account(frame: pd.DataFrame) -> dict[str, float | int]:
    cash = INITIAL_CASH
    count = 0
    for _, trade in frame.sort_values("entry_date").iterrows():
        entry = float(trade["entry_close"])
        exit_price = float(trade["exit_close"])
        quantity = math.floor(cash / (entry * 100)) * 100
        while quantity >= 100 and quantity * entry + fee(quantity * entry) > cash:
            quantity -= 100
        if quantity < 100:
            continue
        buy = quantity * entry
        sell = quantity * exit_price
        cash += sell - fee(sell) - buy - fee(buy)
        count += 1
    return {"initial_cash": INITIAL_CASH, "final_cash": float(cash), "trades": count}


def yearly_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, value in annual_returns(frame).items():
        local = frame[frame["entry_date"].dt.year.eq(year)]
        rows.append(
            {
                "year": int(year),
                "trades": int(len(local)),
                "wins": int((local["ret"] > 0).sum()),
                "win_rate": float((local["ret"] > 0).mean()) if len(local) else np.nan,
                "annual_return": float(value),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = read_baseline()
    d3_closed, d3_open, d_choices = build_d3_history()
    variants: dict[str, pd.DataFrame] = {"baseline": baseline}
    audits: dict[str, pd.DataFrame] = {}
    rows = []
    for mode in ("none", "positive_3", "positive_5", "positive_all"):
        candidate, audit = insert_d3_gaps(baseline, d3_closed, mode)
        variants[mode] = candidate
        audits[mode] = audit
    for name, frame in variants.items():
        all_stat = metrics(frame)
        holdout = metrics(frame, HOLDOUT_YEARS)
        rows.append(
            {
                "variant": name,
                **{f"all_{key}": value for key, value in all_stat.items()},
                **{f"holdout_{key}": value for key, value in holdout.items()},
                "net_final_10000": net_account(frame)["final_cash"],
            }
        )
    table = pd.DataFrame(rows)
    base = table[table["variant"].eq("baseline")].iloc[0]
    candidates = table[~table["variant"].eq("baseline")].copy()
    candidates["passes"] = (
        candidates["all_final_1000"].gt(base["all_final_1000"])
        & candidates["all_win_rate"].ge(base["all_win_rate"])
        & candidates["all_avg_annual_return"].gt(base["all_avg_annual_return"])
        & candidates["holdout_final_1000"].gt(base["holdout_final_1000"])
        & candidates["holdout_win_rate"].ge(base["holdout_win_rate"])
        & candidates["all_max_drawdown"].ge(base["all_max_drawdown"] - 0.01)
    )
    passing = candidates[candidates["passes"]].sort_values(
        ["holdout_final_1000", "all_final_1000"], ascending=False
    )
    promoted = not passing.empty
    winner = str(passing.iloc[0]["variant"]) if promoted else "baseline"
    winner_frame = variants[winner]
    winner_frame.to_csv(OUT / "winner_trades.csv", index=False, encoding="utf-8-sig")
    yearly_table(baseline).to_csv(OUT / "baseline_yearly.csv", index=False, encoding="utf-8-sig")
    yearly_table(winner_frame).to_csv(OUT / "winner_yearly.csv", index=False, encoding="utf-8-sig")
    table.to_csv(OUT / "variants.csv", index=False, encoding="utf-8-sig")
    for name, audit in audits.items():
        audit.to_csv(OUT / f"audit_{name}.csv", index=False, encoding="utf-8-sig")
    d3_closed.to_csv(OUT / "d3_rolling_closed.csv", index=False, encoding="utf-8-sig")
    d3_open.to_csv(OUT / "d3_current_open.csv", index=False, encoding="utf-8-sig")
    d_choices.to_csv(OUT / "d3_choices.csv", index=False, encoding="utf-8-sig")
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "control_variable": "fill residual C3-CS idle windows with D3 only",
        "fixed": "C3-CS trades, single position, T+1 close, baseline preemption and fees",
        "baseline": table[table["variant"].eq("baseline")].iloc[0].to_dict(),
        "winner": table[table["variant"].eq(winner)].iloc[0].to_dict(),
        "winner_variant": winner,
        "passing_variants": int(len(passing)),
        "promoted": promoted,
        "open_d3_excluded_from_closed_performance": int(len(d3_open)),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
