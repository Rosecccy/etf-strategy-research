from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import controlled_upgrade_search as stats_mod
import take_profit_10_test as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "execution_orders"
THRESHOLDS = (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02)
MIN_TRAIN_TRADES = 12

SOURCES = {
    "C": ROOT / "fit" / "rolling_upgrades" / "c_trades.csv",
    "S": ROOT / "fit" / "trend_extension" / "s_trades.csv",
    "D": ROOT / "fit" / "novel_path_exits" / "d_trades.csv",
}


def load_trades(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"symbol": str})
    frame["symbol"] = frame["symbol"].str.zfill(6)
    for column in (
        "entry_date",
        "exit_date_test",
        "control_signal_date",
        "extend_signal_date",
        "path_signal_date",
    ):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def load_prices(project: Path, symbols: set[str]) -> dict[str, pd.DataFrame]:
    result = {}
    for symbol in symbols:
        path = core.raw_path(project, symbol)
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for column in ("open", "high", "low", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        result[symbol] = (
            frame.dropna(subset=["date", "open", "high", "low", "close"])
            .sort_values("date")
            .drop_duplicates("date", keep="last")
            .reset_index(drop=True)
        )
    return result


def order_price(data: pd.DataFrame, date: pd.Timestamp, side: str, threshold: float) -> tuple[float, bool]:
    dates = pd.DatetimeIndex(data["date"])
    pos = int(dates.searchsorted(date, side="left"))
    if pos <= 0 or pos >= len(data) or dates[pos] != date:
        raise ValueError(f"missing execution date {date.date()}")
    row = data.iloc[pos]
    previous_close = float(data.iloc[pos - 1]["close"])
    if side == "buy":
        limit_price = previous_close * (1.0 - threshold)
        if float(row["open"]) <= limit_price:
            return float(row["open"]), True
        if float(row["low"]) <= limit_price:
            return float(limit_price), True
        return float(row["close"]), False
    limit_price = previous_close * (1.0 + threshold)
    if float(row["open"]) >= limit_price:
        return float(row["open"]), True
    if float(row["high"]) >= limit_price:
        return float(limit_price), True
    return float(row["close"]), False


def apply_order(
    trades: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    kind: str,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for _, trade in trades.iterrows():
        item = trade.to_dict()
        data = prices[str(trade["symbol"])]
        item[f"{kind}_order_threshold"] = threshold
        if kind == "entry":
            price, filled = order_price(data, pd.Timestamp(trade["entry_date"]), "buy", threshold)
            item["entry_close"] = price
            item["entry_limit_filled"] = filled
        elif kind == "exit":
            price, filled = order_price(data, pd.Timestamp(trade["exit_date_test"]), "sell", threshold)
            item["exit_close_test"] = price
            item["exit_limit_filled"] = filled
        else:
            raise ValueError(kind)
        item["gross_return_test"] = float(item["exit_close_test"]) / float(item["entry_close"]) - 1.0
        rows.append(item)
    return pd.DataFrame(rows)


def train_stats(frame: pd.DataFrame) -> dict:
    return core.simulate_account(frame)[0]


def choose_threshold(
    candidates: dict[float, pd.DataFrame],
    baseline: pd.DataFrame,
    year: int,
) -> tuple[float | None, dict]:
    train_mask = pd.to_datetime(baseline["entry_date"]).dt.year.lt(year)
    train_base = baseline[train_mask].copy()
    if len(train_base) < MIN_TRAIN_TRADES:
        return None, {"test_year": year, "threshold": np.nan, "reason": "insufficient_history"}
    base = train_stats(train_base)
    rows = []
    for threshold in THRESHOLDS:
        local = candidates[threshold][train_mask].copy()
        current = train_stats(local)
        rows.append(
            {
                "threshold": threshold,
                "ratio": current["final_value"] / base["final_value"],
                "win_delta": current["win_rate"] - base["win_rate"],
                "dd_delta": current["max_drawdown"] - base["max_drawdown"],
            }
        )
    table = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    scored = []
    for index, row in table.iterrows():
        lo = max(0, index - 1)
        hi = min(len(table), index + 2)
        neighborhood = table.iloc[lo:hi]
        stable = bool(
            len(neighborhood) >= 2
            and (neighborhood["ratio"] > 1.0).mean() >= 2 / 3
            and float(neighborhood["ratio"].median()) > 1.002
            and float(neighborhood["win_delta"].median()) >= -1e-12
        )
        score = (
            np.log(max(float(neighborhood["ratio"].median()), 1e-12))
            + 0.7 * float(neighborhood["win_delta"].median())
            + 0.15 * min(float(neighborhood["dd_delta"].median()), 0.10)
        )
        scored.append({**row.to_dict(), "stable": stable, "plateau_score": score})
    stable_rows = [row for row in scored if row["stable"]]
    if not stable_rows:
        return None, {"test_year": year, "threshold": np.nan, "reason": "no_stable_plateau"}
    winner = max(stable_rows, key=lambda row: row["plateau_score"])
    return float(winner["threshold"]), {
        "test_year": year,
        "threshold": float(winner["threshold"]),
        "reason": "past_only_plateau",
        "train_trades": len(train_base),
        "train_ratio": float(winner["ratio"]),
        "train_win_delta": float(winner["win_delta"]),
        "plateau_score": float(winner["plateau_score"]),
    }


def walkforward(
    baseline: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = {threshold: apply_order(baseline, prices, kind, threshold) for threshold in THRESHOLDS}
    years = sorted(pd.to_datetime(baseline["entry_date"]).dt.year.unique())
    parts = []
    selections = []
    for year in years:
        threshold, record = choose_threshold(candidates, baseline, int(year))
        source = baseline if threshold is None else candidates[threshold]
        local = source[pd.to_datetime(source["entry_date"]).dt.year.eq(year)].copy()
        local[f"walkforward_{kind}_threshold"] = np.nan if threshold is None else threshold
        local[f"walkforward_{kind}_policy"] = "close_only" if threshold is None else "limit_then_close"
        parts.append(local)
        selections.append(record)
    return (
        pd.concat(parts, ignore_index=True).sort_values(["entry_date", "symbol"]).reset_index(drop=True),
        pd.DataFrame(selections),
    )


def metrics(frame: pd.DataFrame) -> dict:
    full = stats_mod.subset_stats(frame, "full")
    holdout = stats_mod.subset_stats(frame, "holdout")
    years = pd.to_datetime(frame["entry_date"]).dt.year
    year_count = int(years.max() - years.min() + 1)
    active_years = int(years.nunique())
    return {
        "full": full,
        "holdout": holdout,
        "raw_triggers": int(len(frame)),
        "closed_trades": int(full["closed_trades"]),
        "avg_triggers_per_year": float(len(frame) / year_count),
        "active_year_rate": float(active_years / year_count),
    }


def evaluate_line(line: str, project: Path) -> dict:
    baseline = load_trades(SOURCES[line])
    prices = load_prices(project, set(baseline["symbol"]))
    entry, entry_selection = walkforward(baseline, prices, "entry")
    exit_only, exit_selection = walkforward(baseline, prices, "exit")
    combined, combined_exit_selection = walkforward(entry, prices, "exit")
    variants = {"baseline": baseline, "entry_only": entry, "exit_only": exit_only, "combined": combined}
    base_metrics = metrics(baseline)
    result = {"line": line, "variants": {name: metrics(frame) for name, frame in variants.items()}}
    for name, frame in variants.items():
        value = result["variants"][name]
        value["final_ratio"] = value["full"]["final_value"] / base_metrics["full"]["final_value"]
        value["holdout_ratio"] = value["holdout"]["final_value"] / base_metrics["holdout"]["final_value"]
        frame.to_csv(OUT / f"{line.lower()}_{name}_trades.csv", index=False, encoding="utf-8-sig")
    entry_selection.to_csv(OUT / f"{line.lower()}_entry_selection.csv", index=False, encoding="utf-8-sig")
    exit_selection.to_csv(OUT / f"{line.lower()}_exit_selection.csv", index=False, encoding="utf-8-sig")
    combined_exit_selection.to_csv(
        OUT / f"{line.lower()}_combined_exit_selection.csv", index=False, encoding="utf-8-sig"
    )
    return result


def main() -> None:
    core.configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    projects = {"C": ROOT, "S": ROOT.parent / "S", "D": ROOT}
    results = [evaluate_line(line, projects[line]) for line in ("C", "S", "D")]
    payload = {
        "method": "past-only limit order followed by same-day close completion",
        "thresholds": THRESHOLDS,
        "trigger_constraint": "all baseline triggers must execute; entry and exit dates unchanged",
        "results": results,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
