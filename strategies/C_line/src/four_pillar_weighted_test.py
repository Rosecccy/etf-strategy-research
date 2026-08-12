from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import four_pillar_rolling_test as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "four_pillar" / "weighted"

PILLAR_CHOICES = {
    "volume": ["pillar_volume_reversal", "inv:pillar_volume_reversal", "pillar_volume_momentum", "inv:pillar_volume_momentum"],
    "turnover": ["pillar_turnover", "inv:pillar_turnover"],
    "valuation": ["pillar_valuation", "inv:pillar_valuation"],
    "chip": ["pillar_chip_reversal", "inv:pillar_chip_reversal", "pillar_chip_momentum", "inv:pillar_chip_momentum"],
}

# Every profile keeps all four pillars. The grid is deliberately compact to limit selector overfit.
WEIGHTS = {
    "equal": (0.25, 0.25, 0.25, 0.25),
    "vol": (0.55, 0.25, 0.10, 0.10),
    "turn": (0.25, 0.55, 0.10, 0.10),
    "value": (0.15, 0.15, 0.55, 0.15),
    "chip": (0.15, 0.15, 0.15, 0.55),
    "vol_turn": (0.40, 0.40, 0.10, 0.10),
    "value_chip": (0.10, 0.10, 0.40, 0.40),
    "balanced": (0.35, 0.25, 0.20, 0.20),
}


ORIENTATIONS = {
    "reversal_quiet": ("pillar_volume_reversal", "inv:pillar_turnover", "pillar_valuation", "pillar_chip_reversal"),
    "reversal_active": ("pillar_volume_reversal", "pillar_turnover", "pillar_valuation", "pillar_chip_reversal"),
    "momentum_active": ("pillar_volume_momentum", "pillar_turnover", "pillar_valuation", "pillar_chip_momentum"),
    "momentum_quiet": ("pillar_volume_momentum", "inv:pillar_turnover", "pillar_valuation", "pillar_chip_momentum"),
    "contrarian": ("inv:pillar_volume_momentum", "inv:pillar_turnover", "pillar_valuation", "pillar_chip_reversal"),
    "expensive_momentum": ("pillar_volume_momentum", "pillar_turnover", "inv:pillar_valuation", "pillar_chip_momentum"),
    "cheap_distribution": ("pillar_volume_reversal", "pillar_turnover", "pillar_valuation", "inv:pillar_chip_momentum"),
    "broad_inverse": ("inv:pillar_volume_reversal", "inv:pillar_turnover", "inv:pillar_valuation", "inv:pillar_chip_reversal"),
}


def score_specs():
    for orientation, columns in ORIENTATIONS.items():
        for profile, weights in WEIGHTS.items():
            name = profile + ":" + orientation
            yield name, columns, np.asarray(weights, dtype=float)


def make_score(frame: pd.DataFrame, columns, weights: np.ndarray) -> np.ndarray:
    values = []
    for name in columns:
        inverse = name.startswith("inv:")
        column = name.removeprefix("inv:")
        value = pd.to_numeric(frame[column], errors="coerce").fillna(0.5).clip(0, 1).to_numpy(dtype=float)
        values.append(1.0 - value if inverse else value)
    return np.average(np.vstack(values), axis=0, weights=weights)


def select_year(train: pd.DataFrame, test: pd.DataFrame):
    baseline = core.quick_metrics(train, np.zeros(len(train), dtype=int))
    best = None
    for score_name, columns, weights in score_specs():
        train_score = make_score(train, columns, weights)
        test_score = make_score(test, columns, weights)
        for policy in core.policies()[1:]:
            train_shifts = core.choose_shifts(train_score, policy, train_score)
            metrics = core.quick_metrics(train, train_shifts)
            ratio = metrics["growth"] / baseline["growth"]
            win_delta = metrics["win"] - baseline["win"]
            dd_delta = metrics["dd"] - baseline["dd"]
            if ratio < 0.995 or win_delta < -0.02 or dd_delta < -0.01:
                continue
            objective = float(np.log(max(ratio, 1e-9)) + 0.6 * win_delta + 0.2 * dd_delta)
            if best is None or objective > best[0]:
                best = (objective, score_name, policy, train_score, test_score)
    if best is None:
        return np.zeros(len(test), dtype=int), {"score_name": "baseline", "policy": "baseline", "objective": 0.0}
    objective, score_name, policy, train_score, test_score = best
    shifts = core.choose_shifts(test_score, policy, train_score)
    return shifts, {"score_name": score_name, "policy": policy.key, "objective": objective}


def apply_path(frame: pd.DataFrame, window: str):
    parts, selected_rows = [], []
    years = sorted(frame["entry_date"].dt.year.unique())
    for year in years:
        train = frame[frame["entry_date"].dt.year < year]
        if window != "all":
            train = train[train["entry_date"].dt.year >= year - int(window)]
        test = frame[frame["entry_date"].dt.year == year].copy()
        if len(train) < 8 or train["entry_date"].dt.year.nunique() < 2:
            shifts = np.zeros(len(test), dtype=int)
            selected = {"score_name": "baseline", "policy": "baseline", "objective": 0.0}
        else:
            shifts, selected = select_year(train, test)
        for local_index, (index, row) in enumerate(test.iterrows()):
            shift = int(shifts[local_index])
            test.loc[index, "factor_shift"] = shift
            test.loc[index, "exit_date_test"] = row[f"exit_date_{shift}"]
            test.loc[index, "exit_close_test"] = row[f"exit_close_{shift}"]
            test.loc[index, "gross_return_test"] = row[f"return_{shift}"]
            test.loc[index, "factor_score_name"] = selected["score_name"]
            test.loc[index, "factor_policy"] = selected["policy"]
        selected_rows.append({**selected, "year": year, "window": window, "train_trades": len(train), "test_trades": len(test), "changed": int((shifts != 0).sum())})
        parts.append(test)
    return pd.concat(parts, ignore_index=True).sort_values(["entry_date", "symbol"]), pd.DataFrame(selected_rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(core.FEATURES)
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel["date"])
    rows, all_selected = [], []
    for line in ("C", "S", "D"):
        trades = core.load_trades(line)
        prices = core.load_prices(set(trades["symbol"]))
        frame = core.attach_shift_prices(core.attach_features(trades, panel), prices)
        baseline = core.exact_stats(frame)
        base_dev = core.period_stats(frame, end=2023)
        base_holdout = core.period_stats(frame, start=core.HOLDOUT_START)
        candidates = []
        for window in ("3", "5", "all"):
            path, selected = apply_path(frame, window)
            full = core.exact_stats(path)
            dev = core.period_stats(path, end=2023)
            holdout = core.period_stats(path, start=core.HOLDOUT_START)
            candidates.append((window, path, selected, full, dev, holdout))
        eligible = [x for x in candidates if x[4]["win_rate"] >= base_dev["win_rate"] - 0.02 and x[4]["max_drawdown"] >= base_dev["max_drawdown"] - 0.01]
        chosen = max(eligible or candidates, key=lambda x: (x[4]["final_value"], x[4]["win_rate"], x[3]["final_value"]))
        window, path, selected, full, dev, holdout = chosen
        rows.append({
            "line": line, "window": window, "baseline_final": baseline["final_value"],
            "final_value": full["final_value"], "final_ratio": full["final_value"] / baseline["final_value"],
            "win_rate": full["win_rate"], "avg_annual_return": full["avg_annual_return"],
            "max_drawdown": full["max_drawdown"], "trades": full["trades"],
            "trigger_retention": full["trades"] / baseline["trades"],
            "dev_ratio": dev["final_value"] / base_dev["final_value"],
            "holdout_ratio": holdout["final_value"] / base_holdout["final_value"],
            "holdout_win": holdout["win_rate"], "changed_trades": int(path["factor_shift"].fillna(0).ne(0).sum()),
        })
        path.to_csv(OUT / f"{line.lower()}_trades.csv", index=False, encoding="utf-8-sig")
        all_selected.append(selected.assign(line=line))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "results.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_selected, ignore_index=True).to_csv(OUT / "selected_by_year.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(result.to_dict("records"), ensure_ascii=False, indent=2), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
