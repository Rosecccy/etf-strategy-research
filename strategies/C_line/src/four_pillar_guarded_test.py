from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import four_pillar_rolling_test as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "four_pillar" / "guarded"
GUARDS = {
    "win_guard": {"min_ratio": 0.995, "min_win": 0.0, "min_dd": -0.005, "win_weight": 2.0},
    "strict_gain": {"min_ratio": 1.01, "min_win": 0.0, "min_dd": 0.0, "win_weight": 2.0},
    "win_first": {"min_ratio": 0.99, "min_win": 0.02, "min_dd": -0.005, "win_weight": 3.0},
}


def select_year(train: pd.DataFrame, test: pd.DataFrame, guard: dict):
    baseline = core.quick_metrics(train, np.zeros(len(train), dtype=int))
    best = None
    for score_name, columns in core.score_definitions("all_four").items():
        train_score = core.make_score(train, columns)
        test_score = core.make_score(test, columns)
        for policy in core.policies()[1:]:
            train_shifts = core.choose_shifts(train_score, policy, train_score)
            metrics = core.quick_metrics(train, train_shifts)
            ratio = metrics["growth"] / baseline["growth"]
            win_delta = metrics["win"] - baseline["win"]
            dd_delta = metrics["dd"] - baseline["dd"]
            if ratio < guard["min_ratio"] or win_delta < guard["min_win"] or dd_delta < guard["min_dd"]:
                continue
            objective = float(np.log(max(ratio, 1e-9)) + guard["win_weight"] * win_delta + 0.3 * dd_delta)
            if best is None or objective > best[0]:
                best = (objective, score_name, policy, train_score, test_score)
    if best is None:
        return np.zeros(len(test), dtype=int), {"score_name": "baseline", "policy": "baseline", "objective": 0.0}
    objective, name, policy, train_score, test_score = best
    return core.choose_shifts(test_score, policy, train_score), {"score_name": name, "policy": policy.key, "objective": objective}


def apply_path(frame: pd.DataFrame, window: str, guard_name: str):
    parts, selections = [], []
    for year in sorted(frame["entry_date"].dt.year.unique()):
        train = frame[frame["entry_date"].dt.year < year]
        if window != "all":
            train = train[train["entry_date"].dt.year >= year - int(window)]
        test = frame[frame["entry_date"].dt.year == year].copy()
        if len(train) < 12 or train["entry_date"].dt.year.nunique() < 3:
            shifts = np.zeros(len(test), dtype=int)
            selected = {"score_name": "baseline", "policy": "baseline", "objective": 0.0}
        else:
            shifts, selected = select_year(train, test, GUARDS[guard_name])
        for local_index, (index, row) in enumerate(test.iterrows()):
            shift = int(shifts[local_index])
            test.loc[index, "factor_shift"] = shift
            test.loc[index, "exit_date_test"] = row[f"exit_date_{shift}"]
            test.loc[index, "exit_close_test"] = row[f"exit_close_{shift}"]
            test.loc[index, "gross_return_test"] = row[f"return_{shift}"]
            test.loc[index, "factor_score_name"] = selected["score_name"]
            test.loc[index, "factor_policy"] = selected["policy"]
        selections.append({**selected, "year": year, "guard": guard_name, "window": window, "train_trades": len(train), "test_trades": len(test), "changed": int((shifts != 0).sum())})
        parts.append(test)
    return pd.concat(parts, ignore_index=True).sort_values(["entry_date", "symbol"]), pd.DataFrame(selections)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(core.FEATURES)
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel["date"])
    rows, selected_frames = [], []
    for line in ("C", "D"):
        trades = core.load_trades(line)
        frame = core.attach_shift_prices(core.attach_features(trades, panel), core.load_prices(set(trades["symbol"])))
        baseline = core.exact_stats(frame)
        base_dev = core.period_stats(frame, end=2023)
        base_holdout = core.period_stats(frame, start=core.HOLDOUT_START)
        candidates = []
        for guard in GUARDS:
            for window in ("3", "5", "all"):
                path, selections = apply_path(frame, window, guard)
                full, dev = core.exact_stats(path), core.period_stats(path, end=2023)
                holdout = core.period_stats(path, start=core.HOLDOUT_START)
                row = {
                    "line": line, "guard": guard, "window": window,
                    "baseline_final": baseline["final_value"], "final_value": full["final_value"],
                    "final_ratio": full["final_value"] / baseline["final_value"], "win_rate": full["win_rate"],
                    "avg_annual_return": full["avg_annual_return"], "max_drawdown": full["max_drawdown"],
                    "trades": full["trades"], "trigger_retention": full["trades"] / baseline["trades"],
                    "dev_ratio": dev["final_value"] / base_dev["final_value"],
                    "holdout_ratio": holdout["final_value"] / base_holdout["final_value"], "holdout_win": holdout["win_rate"],
                    "changed_trades": int(path["factor_shift"].fillna(0).ne(0).sum()),
                }
                rows.append(row)
                candidates.append((row, path, selections))
        eligible = [
            x for x in candidates
            if x[0]["dev_ratio"] >= 1.0
            and x[0]["final_ratio"] >= 1.0
            and x[0]["holdout_ratio"] >= 0.99
            and x[0]["win_rate"] >= baseline["win_rate"]
            and x[0]["max_drawdown"] >= baseline["max_drawdown"] - 0.005
        ]
        if eligible:
            best = max(eligible, key=lambda x: (x[0]["dev_ratio"], x[0]["win_rate"], x[0]["final_ratio"]))
            best_path, best_selection = best[1], best[2].assign(line=line, chosen=True)
        else:
            best_path = frame.copy()
            best_path["factor_shift"] = 0
            best_path["factor_score_name"] = "baseline"
            best_path["factor_policy"] = "baseline"
            best_selection = pd.DataFrame([{
                "line": line, "chosen": True, "year": "all", "guard": "baseline",
                "window": "baseline", "score_name": "baseline", "policy": "baseline",
                "objective": 0.0, "changed": 0, "train_trades": len(frame), "test_trades": len(frame),
            }])
        best_path.to_csv(OUT / f"{line.lower()}_best_trades.csv", index=False, encoding="utf-8-sig")
        selected_frames.append(best_selection)
    result = pd.DataFrame(rows).sort_values(["line", "final_ratio"], ascending=[True, False])
    result.to_csv(OUT / "results.csv", index=False, encoding="utf-8-sig")
    pd.concat(selected_frames, ignore_index=True).to_csv(OUT / "best_selected_by_year.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(result.to_dict("records"), ensure_ascii=False, indent=2), encoding="utf-8")
    print(result.groupby("line", as_index=False).head(4).to_string(index=False))


if __name__ == "__main__":
    main()
