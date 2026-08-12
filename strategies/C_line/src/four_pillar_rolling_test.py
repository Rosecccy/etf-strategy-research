from __future__ import annotations

import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import take_profit_10_test as account  # noqa: E402


FEATURES = ROOT / "factors" / "panel.parquet"
TRADE_ROOT = ROOT / "fit" / "trigger_safe_upgrades"
OUT = ROOT / "fit" / "four_pillar"
PRICE_DIR = ROOT / "raw" / "etf"
S_PRICE_DIR = ROOT.parent / "S" / "raw" / "etf"
SHIFTS = (-5, -3, 0, 3, 5, 8, 10)
HOLDOUT_START = 2024


@dataclass(frozen=True)
class Policy:
    kind: str
    low_q: float | None = None
    high_q: float | None = None
    low_shift: int = 0
    high_shift: int = 0

    @property
    def key(self) -> str:
        return f"{self.kind}|lq={self.low_q}|hq={self.high_q}|ls={self.low_shift}|hs={self.high_shift}"


def policies() -> list[Policy]:
    result = [Policy("baseline")]
    result += [Policy("high", high_q=q, high_shift=shift) for q in (0.55, 0.65, 0.75) for shift in (3, 5, 8, 10)]
    result += [Policy("low", low_q=q, low_shift=-shift) for q in (0.25, 0.35, 0.45) for shift in (3, 5)]
    result += [
        Policy("both", low_q=lq, high_q=hq, low_shift=-cut, high_shift=extend)
        for lq, hq, cut, extend in itertools.product((0.30, 0.40), (0.60, 0.70), (3, 5), (3, 5, 8))
    ]
    return result


def load_prices(symbols: set[str]) -> dict[str, pd.DataFrame]:
    prices = {}
    for symbol in symbols:
        path = PRICE_DIR / f"{symbol}.csv"
        if not path.exists():
            path = S_PRICE_DIR / f"{symbol}.csv"
        data = pd.read_csv(path)
        data["date"] = pd.to_datetime(data["date"])
        prices[symbol] = data.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return prices


def load_trades(line: str) -> pd.DataFrame:
    data = pd.read_csv(TRADE_ROOT / f"{line.lower()}_trades.csv", dtype={"symbol": str})
    data["symbol"] = data["symbol"].str.zfill(6)
    for column in ("entry_date", "exit_date_test"):
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data = data.sort_values(["entry_date", "symbol"]).reset_index(drop=True)
    data["line"] = line
    return data


def attach_features(trades: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    left = trades.sort_values(["symbol", "entry_date"]).copy()
    right_columns = [
        "symbol",
        "date",
        "pillar_volume_reversal",
        "pillar_volume_momentum",
        "pillar_turnover",
        "pillar_valuation",
        "pillar_chip_reversal",
        "pillar_chip_momentum",
        "volume_ratio20",
        "turnover_rate",
        "chip_profit60",
        "valuation_cheap",
    ]
    right = panel[right_columns].sort_values(["symbol", "date"])
    merged_parts = []
    for symbol, group in left.groupby("symbol", sort=False):
        local = right[right["symbol"].eq(symbol)].drop(columns="symbol")
        merged = pd.merge_asof(
            group.sort_values("entry_date"),
            local.sort_values("date"),
            left_on="entry_date",
            right_on="date",
            direction="backward",
            allow_exact_matches=False,
        )
        merged_parts.append(merged)
    result = pd.concat(merged_parts, ignore_index=True).sort_values(["entry_date", "symbol"]).reset_index(drop=True)
    result = result.rename(columns={"date": "factor_date"})
    return result


def shifted_exit(data: pd.DataFrame, entry_date: pd.Timestamp, exit_date: pd.Timestamp, next_entry: pd.Timestamp | None, shift: int) -> tuple[pd.Timestamp, float]:
    dates = pd.DatetimeIndex(data["date"])
    entry_pos = int(dates.searchsorted(entry_date, side="left"))
    exit_pos = int(dates.searchsorted(exit_date, side="left"))
    exit_pos = min(max(exit_pos, 0), len(data) - 1)
    target = min(max(exit_pos + shift, entry_pos + 5), len(data) - 1)
    if next_entry is not None and target > exit_pos:
        cap = int(dates.searchsorted(next_entry, side="right") - 1)
        target = min(target, max(cap, exit_pos))
    return pd.Timestamp(data.iloc[target]["date"]), float(data.iloc[target]["close"])


def attach_shift_prices(trades: pd.DataFrame, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    result = trades.copy()
    for index, row in result.iterrows():
        next_entry = pd.Timestamp(result.iloc[index + 1]["entry_date"]) if index + 1 < len(result) else None
        data = prices[str(row["symbol"])]
        for shift in SHIFTS:
            date, close = shifted_exit(data, pd.Timestamp(row["entry_date"]), pd.Timestamp(row["exit_date_test"]), next_entry, shift)
            result.loc[index, f"exit_date_{shift}"] = date
            result.loc[index, f"exit_close_{shift}"] = close
            result.loc[index, f"return_{shift}"] = close / float(row["entry_close"]) - 1.0
    return result


def score_definitions(stage: str) -> dict[str, list[str]]:
    choices = {
        "volume": ["pillar_volume_reversal", "inv:pillar_volume_reversal", "pillar_volume_momentum", "inv:pillar_volume_momentum"],
        "turnover": ["pillar_turnover", "inv:pillar_turnover"],
        "valuation": ["pillar_valuation", "inv:pillar_valuation"],
        "chip": ["pillar_chip_reversal", "inv:pillar_chip_reversal", "pillar_chip_momentum", "inv:pillar_chip_momentum"],
    }
    stages = {
        "volume": ["volume"],
        "turnover": ["turnover"],
        "valuation": ["valuation"],
        "chip": ["chip"],
        "volume_turnover": ["volume", "turnover"],
        "volume_turnover_valuation": ["volume", "turnover", "valuation"],
        "all_four": ["volume", "turnover", "valuation", "chip"],
    }
    names = stages[stage]
    return {"+".join(combo): list(combo) for combo in itertools.product(*(choices[name] for name in names))}


def make_score(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = []
    for name in columns:
        inverse = name.startswith("inv:")
        column = name.removeprefix("inv:")
        value = pd.to_numeric(frame[column], errors="coerce").fillna(0.5).clip(0, 1).to_numpy(dtype=float)
        values.append(1.0 - value if inverse else value)
    return np.mean(np.vstack(values), axis=0)


def choose_shifts(score: np.ndarray, policy: Policy, train_score: np.ndarray) -> np.ndarray:
    selected = np.zeros(len(score), dtype=int)
    if policy.kind == "baseline":
        return selected
    low = np.nanquantile(train_score, policy.low_q) if policy.low_q is not None else None
    high = np.nanquantile(train_score, policy.high_q) if policy.high_q is not None else None
    if low is not None:
        selected[score <= low] = policy.low_shift
    if high is not None:
        selected[score >= high] = policy.high_shift
    return selected


def quick_metrics(frame: pd.DataFrame, shifts: np.ndarray) -> dict:
    returns = np.empty(len(frame), dtype=float)
    for shift in np.unique(shifts):
        mask = shifts == shift
        returns[mask] = pd.to_numeric(frame.loc[mask, f"return_{int(shift)}"], errors="coerce").to_numpy(dtype=float)
    equity = np.cumprod(1.0 + returns)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    return {
        "growth": float(equity[-1]) if len(equity) else 1.0,
        "win": float((returns > 0).mean()) if len(returns) else 0.0,
        "dd": float(drawdown.min()) if len(drawdown) else 0.0,
    }


def select_for_year(train: pd.DataFrame, test: pd.DataFrame, stage: str) -> tuple[np.ndarray, dict]:
    baseline = quick_metrics(train, np.zeros(len(train), dtype=int))
    best = {"objective": 0.0, "score_name": "baseline", "policy": Policy("baseline"), "train": baseline}
    for score_name, columns in score_definitions(stage).items():
        train_score = make_score(train, columns)
        test_score = make_score(test, columns)
        for policy in policies()[1:]:
            train_shifts = choose_shifts(train_score, policy, train_score)
            metrics = quick_metrics(train, train_shifts)
            ratio = metrics["growth"] / baseline["growth"]
            win_delta = metrics["win"] - baseline["win"]
            dd_delta = metrics["dd"] - baseline["dd"]
            if ratio < 0.995 or win_delta < -0.02 or dd_delta < -0.01:
                continue
            objective = float(np.log(max(ratio, 1e-9)) + 0.6 * win_delta + 0.2 * dd_delta)
            if objective > best["objective"]:
                best = {"objective": objective, "score_name": score_name, "policy": policy, "train": metrics}
                best["train_threshold_score"] = train_score
                best["test_score"] = test_score
    if best["score_name"] == "baseline":
        shifts = np.zeros(len(test), dtype=int)
    else:
        shifts = choose_shifts(best["test_score"], best["policy"], best["train_threshold_score"])
    selected = {
        "score_name": best["score_name"],
        "policy": best["policy"].key,
        "objective": best["objective"],
        "changed": int((shifts != 0).sum()),
    }
    return shifts, selected


def apply_path(frame: pd.DataFrame, stage: str, window: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(pd.to_datetime(frame["entry_date"]).dt.year.unique())
    parts = []
    selections = []
    for year in years:
        all_prior = frame[pd.to_datetime(frame["entry_date"]).dt.year < year]
        if window != "all":
            all_prior = all_prior[pd.to_datetime(all_prior["entry_date"]).dt.year >= year - int(window)]
        test = frame[pd.to_datetime(frame["entry_date"]).dt.year == year].copy()
        if len(all_prior) < 8 or pd.to_datetime(all_prior["entry_date"]).dt.year.nunique() < 2:
            shifts = np.zeros(len(test), dtype=int)
            selected = {"score_name": "baseline", "policy": "baseline", "objective": 0.0, "changed": 0}
        else:
            shifts, selected = select_for_year(all_prior, test, stage)
        for local_index, (index, row) in enumerate(test.iterrows()):
            shift = int(shifts[local_index])
            test.loc[index, "factor_shift"] = shift
            test.loc[index, "exit_date_test"] = row[f"exit_date_{shift}"]
            test.loc[index, "exit_close_test"] = row[f"exit_close_{shift}"]
            test.loc[index, "gross_return_test"] = row[f"return_{shift}"]
            test.loc[index, "factor_stage"] = stage
            test.loc[index, "factor_score_name"] = selected["score_name"]
            test.loc[index, "factor_policy"] = selected["policy"]
        selected.update({"year": year, "stage": stage, "window": window, "train_trades": len(all_prior), "test_trades": len(test)})
        selections.append(selected)
        parts.append(test)
    return pd.concat(parts, ignore_index=True).sort_values(["entry_date", "symbol"]), pd.DataFrame(selections)


def exact_stats(frame: pd.DataFrame) -> dict:
    stats, _, _ = account.simulate_account(frame)
    return stats


def period_stats(frame: pd.DataFrame, start: int | None = None, end: int | None = None) -> dict:
    years = pd.to_datetime(frame["entry_date"]).dt.year
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= years >= start
    if end is not None:
        mask &= years <= end
    return exact_stats(frame[mask].copy())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(FEATURES)
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel["date"])
    stages = ["volume", "turnover", "valuation", "chip", "volume_turnover", "volume_turnover_valuation", "all_four"]
    summary_rows = []
    best_paths = {}
    all_selections = []
    coverage_rows = []
    for line in ("C", "S", "D"):
        trades = load_trades(line)
        prices = load_prices(set(trades["symbol"]))
        attached = attach_features(trades, panel)
        attached = attach_shift_prices(attached, prices)
        coverage_rows.append({
            "line": line,
            "trades": len(attached),
            "volume": float(attached["pillar_volume_reversal"].notna().mean()),
            "turnover": float(attached["pillar_turnover"].notna().mean()),
            "valuation": float(attached["pillar_valuation"].notna().mean()),
            "chip": float(attached["pillar_chip_reversal"].notna().mean()),
        })
        baseline = exact_stats(attached)
        for stage in stages:
            candidates = []
            for window in ("3", "5", "all"):
                path, selections = apply_path(attached, stage, window)
                full = exact_stats(path)
                dev = period_stats(path, end=2023)
                holdout = period_stats(path, start=HOLDOUT_START)
                candidates.append((window, path, selections, full, dev, holdout))
            base_dev = period_stats(attached, end=2023)
            eligible = [item for item in candidates if item[4]["win_rate"] >= base_dev["win_rate"] - 0.02 and item[4]["max_drawdown"] >= base_dev["max_drawdown"] - 0.01]
            chosen = max(eligible or candidates, key=lambda item: (item[4]["final_value"], item[4]["win_rate"], item[3]["final_value"]))
            window, path, selections, full, dev, holdout = chosen
            base_holdout = period_stats(attached, start=HOLDOUT_START)
            summary_rows.append({
                "line": line,
                "stage": stage,
                "window": window,
                "baseline_final": baseline["final_value"],
                "final_value": full["final_value"],
                "final_ratio": full["final_value"] / baseline["final_value"],
                "win_rate": full["win_rate"],
                "avg_annual_return": full["avg_annual_return"],
                "max_drawdown": full["max_drawdown"],
                "trades": full["trades"],
                "trigger_retention": full["trades"] / baseline["trades"],
                "dev_final": dev["final_value"],
                "dev_ratio": dev["final_value"] / base_dev["final_value"],
                "holdout_final": holdout["final_value"],
                "holdout_ratio": holdout["final_value"] / base_holdout["final_value"],
                "holdout_win": holdout["win_rate"],
                "changed_trades": int(pd.to_numeric(path["factor_shift"], errors="coerce").fillna(0).ne(0).sum()),
            })
            all_selections.append(selections.assign(line=line))
            if stage == "all_four":
                best_paths[line] = path

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "stage_results.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_selections, ignore_index=True).to_csv(OUT / "selected_by_year.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coverage_rows).to_csv(OUT / "trade_feature_coverage.csv", index=False, encoding="utf-8-sig")
    for line, path in best_paths.items():
        path.to_csv(OUT / f"{line.lower()}_all_four_trades.csv", index=False, encoding="utf-8-sig")
    payload = {
        "method": "strict annual past-only exit timing overlay; all entry triggers preserved",
        "stages": stages,
        "results": summary.to_dict(orient="records"),
        "coverage": coverage_rows,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
