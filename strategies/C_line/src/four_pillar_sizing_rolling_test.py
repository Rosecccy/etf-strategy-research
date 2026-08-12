from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import four_pillar_rolling_test as base


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "four_pillar" / "sizing"


def policies() -> list[tuple[float, float]]:
    return list(itertools.product((0.20, 0.30, 0.40, 0.50), (0.25, 0.50, 0.75)))


def returns(frame: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(frame["gross_return_test"], errors="coerce").fillna(0).to_numpy(dtype=float)


def quick_metrics(frame: pd.DataFrame, weights: np.ndarray) -> dict:
    ret = returns(frame) * weights
    equity = np.cumprod(1.0 + ret)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    return {
        "growth": float(equity[-1]) if len(equity) else 1.0,
        "win": float((ret > 0).mean()) if len(ret) else 0.0,
        "dd": float(drawdown.min()) if len(drawdown) else 0.0,
    }


def choose(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, dict]:
    baseline = quick_metrics(train, np.ones(len(train)))
    best = {"objective": 0.0, "score_name": "baseline", "q": 0.0, "floor": 1.0, "threshold": np.nan}
    for score_name, columns in base.score_definitions("all_four").items():
        train_score = base.make_score(train, columns)
        for q, floor in policies():
            threshold = float(np.nanquantile(train_score, q))
            train_weights = np.where(train_score <= threshold, floor, 1.0)
            metrics = quick_metrics(train, train_weights)
            ratio = metrics["growth"] / baseline["growth"]
            dd_delta = metrics["dd"] - baseline["dd"]
            if ratio < 0.995 or dd_delta < -0.005:
                continue
            objective = float(np.log(max(ratio, 1e-9)) + 0.3 * dd_delta)
            if objective > best["objective"]:
                best = {"objective": objective, "score_name": score_name, "q": q, "floor": floor, "threshold": threshold}
    if best["score_name"] == "baseline":
        return np.ones(len(test)), best
    score = base.make_score(test, base.score_definitions("all_four")[best["score_name"]])
    return np.where(score <= best["threshold"], best["floor"], 1.0), best


def simulate(frame: pd.DataFrame) -> dict:
    cash = 10_000.0
    account_returns = []
    wins = []
    for _, trade in frame.sort_values(["entry_date", "symbol"]).iterrows():
        before = cash
        weight = float(trade.get("factor_weight", 1.0))
        entry = float(trade["entry_close"])
        exit_price = float(trade["exit_close_test"])
        budget = cash * weight
        quantity = math.floor(budget / (entry * 100)) * 100
        while quantity >= 100 and quantity * entry + base.account.commission(quantity * entry) > budget:
            quantity -= 100
        if quantity < 100:
            quantity = 0
        buy_value = quantity * entry
        buy_fee = base.account.commission(buy_value) if quantity else 0.0
        cash -= buy_value + buy_fee
        sell_value = quantity * exit_price
        sell_fee = base.account.commission(sell_value) if quantity else 0.0
        cash += sell_value - sell_fee
        account_return = cash / before - 1.0
        account_returns.append(account_return)
        wins.append(account_return > 0)
    equity = 10_000.0 * np.cumprod(1.0 + np.asarray(account_returns))
    drawdown = equity / np.maximum.accumulate(equity) - 1.0 if len(equity) else np.array([0.0])
    years = pd.to_datetime(frame["entry_date"]).dt.year
    annual = []
    for year in range(int(years.min()), 2027):
        mask = years.eq(year).to_numpy()
        annual.append(float(np.prod(1.0 + np.asarray(account_returns)[mask]) - 1.0))
    return {
        "final_value": float(cash),
        "win_rate": float(np.mean(wins)) if wins else 0.0,
        "avg_annual_return": float(np.mean(annual)),
        "max_drawdown": float(drawdown.min()),
        "trades": len(frame),
    }


def apply_path(frame: pd.DataFrame, window: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    selected_rows = []
    years = sorted(pd.to_datetime(frame["entry_date"]).dt.year.unique())
    for year in years:
        train = frame[pd.to_datetime(frame["entry_date"]).dt.year < year]
        if window != "all":
            train = train[pd.to_datetime(train["entry_date"]).dt.year >= year - int(window)]
        test = frame[pd.to_datetime(frame["entry_date"]).dt.year == year].copy()
        if len(train) < 8 or pd.to_datetime(train["entry_date"]).dt.year.nunique() < 2:
            weights = np.ones(len(test))
            chosen = {"objective": 0.0, "score_name": "baseline", "q": 0.0, "floor": 1.0, "threshold": np.nan}
        else:
            weights, chosen = choose(train, test)
        test["factor_weight"] = weights
        chosen.update({"year": year, "window": window, "train_trades": len(train), "test_trades": len(test), "reduced": int((weights < 1).sum())})
        selected_rows.append(chosen)
        parts.append(test)
    return pd.concat(parts, ignore_index=True).sort_values(["entry_date", "symbol"]), pd.DataFrame(selected_rows)


def subset(frame: pd.DataFrame, start: int | None = None, end: int | None = None) -> pd.DataFrame:
    years = pd.to_datetime(frame["entry_date"]).dt.year
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= years >= start
    if end is not None:
        mask &= years <= end
    return frame[mask].copy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(base.FEATURES)
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel["date"])
    rows = []
    selections = []
    for line in ("C", "S", "D"):
        trades = base.attach_features(base.load_trades(line), panel)
        trades["factor_weight"] = 1.0
        baseline = simulate(trades)
        base_dev = simulate(subset(trades, end=2023))
        base_holdout = simulate(subset(trades, start=2024))
        candidates = []
        for window in ("3", "5", "all"):
            path, selected = apply_path(trades, window)
            full = simulate(path)
            dev = simulate(subset(path, end=2023))
            holdout = simulate(subset(path, start=2024))
            candidates.append((window, path, selected, full, dev, holdout))
        eligible = [item for item in candidates if item[4]["final_value"] >= base_dev["final_value"] * 0.995 and item[4]["max_drawdown"] >= base_dev["max_drawdown"] - 0.005]
        chosen = max(eligible or candidates, key=lambda item: (item[4]["final_value"], item[4]["max_drawdown"], item[3]["final_value"]))
        window, path, selected, full, dev, holdout = chosen
        selected["line"] = line
        selections.append(selected)
        path.to_csv(OUT / f"{line}_trades.csv", index=False, encoding="utf-8-sig")
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
            "reduced_trades": int(path["factor_weight"].lt(1).sum()),
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "results.csv", index=False, encoding="utf-8-sig")
    pd.concat(selections, ignore_index=True).to_csv(OUT / "selected_by_year.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(result.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
