from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import four_pillar_rolling_test as base


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "four_pillar" / "entry_timing"
DELAYS = (0, 1, 2, 3)


def attach_entry_prices(trades: pd.DataFrame, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    result = trades.copy()
    for index, row in result.iterrows():
        data = prices[str(row["symbol"])]
        dates = pd.DatetimeIndex(data["date"])
        entry_pos = int(dates.searchsorted(pd.Timestamp(row["entry_date"]), side="left"))
        exit_pos = int(dates.searchsorted(pd.Timestamp(row["exit_date_test"]), side="left"))
        for delay in DELAYS:
            target = min(entry_pos + delay, max(entry_pos, exit_pos - 5))
            result.loc[index, f"entry_date_{delay}"] = pd.Timestamp(data.iloc[target]["date"])
            result.loc[index, f"entry_close_{delay}"] = float(data.iloc[target]["close"])
            result.loc[index, f"entry_return_{delay}"] = float(row["exit_close_test"]) / float(data.iloc[target]["close"]) - 1.0
    return result


def quick_metrics(frame: pd.DataFrame, delays: np.ndarray) -> dict:
    returns = np.empty(len(frame), dtype=float)
    for delay in np.unique(delays):
        mask = delays == delay
        returns[mask] = pd.to_numeric(frame.loc[mask, f"entry_return_{int(delay)}"], errors="coerce").to_numpy(dtype=float)
    equity = np.cumprod(1.0 + returns)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    return {
        "growth": float(equity[-1]) if len(equity) else 1.0,
        "win": float((returns > 0).mean()) if len(returns) else 0.0,
        "dd": float(drawdown.min()) if len(drawdown) else 0.0,
    }


def policies() -> list[tuple[str, float, int]]:
    return [(side, q, delay) for side, q, delay in itertools.product(("high", "low"), (0.25, 0.35, 0.45, 0.55, 0.65, 0.75), (1, 2, 3))]


def choose(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, dict]:
    baseline = quick_metrics(train, np.zeros(len(train), dtype=int))
    best = {"objective": 0.0, "score_name": "baseline", "side": "", "q": 0.0, "delay": 0, "threshold": np.nan}
    for score_name, columns in base.score_definitions("all_four").items():
        train_score = base.make_score(train, columns)
        for side, q, delay in policies():
            threshold = float(np.nanquantile(train_score, q))
            mask = train_score >= threshold if side == "high" else train_score <= threshold
            train_delays = np.where(mask, delay, 0)
            metrics = quick_metrics(train, train_delays)
            ratio = metrics["growth"] / baseline["growth"]
            win_delta = metrics["win"] - baseline["win"]
            dd_delta = metrics["dd"] - baseline["dd"]
            if ratio < 0.995 or win_delta < -0.02 or dd_delta < -0.01:
                continue
            objective = float(np.log(max(ratio, 1e-9)) + 0.6 * win_delta + 0.2 * dd_delta)
            if objective > best["objective"]:
                best = {"objective": objective, "score_name": score_name, "side": side, "q": q, "delay": delay, "threshold": threshold}
    if best["score_name"] == "baseline":
        return np.zeros(len(test), dtype=int), best
    score = base.make_score(test, base.score_definitions("all_four")[best["score_name"]])
    mask = score >= best["threshold"] if best["side"] == "high" else score <= best["threshold"]
    return np.where(mask, best["delay"], 0), best


def apply_path(frame: pd.DataFrame, window: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(pd.to_datetime(frame["entry_date"]).dt.year.unique())
    parts = []
    selected_rows = []
    for year in years:
        train = frame[pd.to_datetime(frame["entry_date"]).dt.year < year]
        if window != "all":
            train = train[pd.to_datetime(train["entry_date"]).dt.year >= year - int(window)]
        test = frame[pd.to_datetime(frame["entry_date"]).dt.year == year].copy()
        if len(train) < 8 or pd.to_datetime(train["entry_date"]).dt.year.nunique() < 2:
            delays = np.zeros(len(test), dtype=int)
            chosen = {"objective": 0.0, "score_name": "baseline", "side": "", "q": 0.0, "delay": 0, "threshold": np.nan}
        else:
            delays, chosen = choose(train, test)
        for local, (index, row) in enumerate(test.iterrows()):
            delay = int(delays[local])
            test.loc[index, "factor_entry_delay"] = delay
            test.loc[index, "entry_date"] = row[f"entry_date_{delay}"]
            test.loc[index, "entry_close"] = row[f"entry_close_{delay}"]
            test.loc[index, "gross_return_test"] = row[f"entry_return_{delay}"]
        chosen.update({"year": year, "window": window, "train_trades": len(train), "test_trades": len(test), "changed": int((delays > 0).sum())})
        selected_rows.append(chosen)
        parts.append(test)
    return pd.concat(parts, ignore_index=True).sort_values(["entry_date", "symbol"]), pd.DataFrame(selected_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(base.FEATURES)
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel["date"])
    rows = []
    selections = []
    for line in ("C", "S", "D"):
        trades = base.load_trades(line)
        prices = base.load_prices(set(trades["symbol"]))
        attached = base.attach_features(trades, panel)
        attached = attach_entry_prices(attached, prices)
        baseline = base.exact_stats(attached)
        base_dev = base.period_stats(attached, end=2023)
        base_holdout = base.period_stats(attached, start=base.HOLDOUT_START)
        paths = []
        for window in ("3", "5", "all"):
            path, selected = apply_path(attached, window)
            full = base.exact_stats(path)
            dev = base.period_stats(path, end=2023)
            holdout = base.period_stats(path, start=base.HOLDOUT_START)
            paths.append((window, path, selected, full, dev, holdout))
        eligible = [item for item in paths if item[4]["win_rate"] >= base_dev["win_rate"] - 0.02 and item[4]["max_drawdown"] >= base_dev["max_drawdown"] - 0.01]
        chosen = max(eligible or paths, key=lambda item: (item[4]["final_value"], item[4]["win_rate"], item[3]["final_value"]))
        window, path, selected, full, dev, holdout = chosen
        selected["line"] = line
        selections.append(selected)
        path.to_csv(OUT / f"{line.lower()}_trades.csv", index=False, encoding="utf-8-sig")
        rows.append({
            "line": line,
            "window": window,
            "baseline_final": baseline["final_value"],
            "final_value": full["final_value"],
            "final_ratio": full["final_value"] / baseline["final_value"],
            "win_rate": full["win_rate"],
            "avg_annual_return": full["avg_annual_return"],
            "max_drawdown": full["max_drawdown"],
            "trades": full["trades"],
            "trigger_retention": full["trades"] / baseline["trades"],
            "dev_ratio": dev["final_value"] / base_dev["final_value"],
            "holdout_ratio": holdout["final_value"] / base_holdout["final_value"],
            "holdout_win": holdout["win_rate"],
            "changed_trades": int(pd.to_numeric(path["factor_entry_delay"], errors="coerce").fillna(0).gt(0).sum()),
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "results.csv", index=False, encoding="utf-8-sig")
    pd.concat(selections, ignore_index=True).to_csv(OUT / "selected_by_year.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(result.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
