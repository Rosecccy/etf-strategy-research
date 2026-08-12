from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import controlled_upgrade_search as search
import take_profit_10_test as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "novel_path_exits"
MIN_TRAIN_TRADES = 12


@dataclass(frozen=True)
class PathRule:
    family: str
    hold: int
    a: float
    b: float
    window: int = 0

    @property
    def key(self) -> str:
        return f"{self.family}:h{self.hold}:a{self.a:+.3f}:b{self.b:+.3f}:w{self.window}"


def rules() -> list[PathRule]:
    result = []
    result += [
        PathRule("stale_profit", hold, peak, floor)
        for hold in (5, 10, 15, 20, 30)
        for peak in (0.03, 0.05, 0.08, 0.10, 0.15)
        for floor in (0.00, 0.01, 0.02, 0.03, 0.05)
        if floor < peak
    ]
    result += [
        PathRule("no_progress", hold, max_peak, floor)
        for hold in (5, 10, 15, 20, 30)
        for max_peak in (0.01, 0.02, 0.03, 0.05, 0.08)
        for floor in (-0.05, -0.03, -0.02, 0.00, 0.01)
    ]
    result += [
        PathRule("underwater", hold, fraction, floor, window)
        for hold in (10, 15, 20, 30)
        for window in (5, 10, 15, 20)
        for fraction in (0.60, 0.70, 0.80, 0.90)
        for floor in (-0.05, -0.03, 0.00)
        if window <= hold
    ]
    return result


def apply_rule(
    frame: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    rule: PathRule,
) -> pd.DataFrame:
    rows = []
    for _, trade in frame.iterrows():
        item = trade.to_dict()
        data = prices[str(trade["symbol"])]
        dates = pd.DatetimeIndex(data["date"])
        closes = data["close"].to_numpy(dtype=float)
        entry_date = pd.Timestamp(trade["entry_date"])
        exit_date = pd.Timestamp(trade["exit_date_test"])
        entry_pos = int(dates.searchsorted(entry_date, side="left"))
        exit_pos = min(int(dates.searchsorted(exit_date, side="left")), len(data) - 1)
        entry_price = float(trade["entry_close"])
        peak_return = -np.inf
        triggered = False
        item["path_rule"] = rule.key
        item["path_triggered"] = False
        item["path_signal_date"] = pd.NaT
        for signal_pos in range(entry_pos + 1, exit_pos):
            current_return = closes[signal_pos] / entry_price - 1.0
            peak_return = max(peak_return, current_return)
            elapsed = signal_pos - entry_pos
            if elapsed < rule.hold:
                continue
            if rule.family == "stale_profit":
                hit = peak_return >= rule.a and rule.b <= current_return <= min(rule.a, peak_return - 0.01)
            elif rule.family == "no_progress":
                hit = peak_return <= rule.a and current_return <= rule.b
            elif rule.family == "underwater":
                start = max(entry_pos + 1, signal_pos - rule.window + 1)
                underwater = float(np.mean(closes[start:signal_pos + 1] < entry_price))
                hit = underwater >= rule.a and current_return <= rule.b
            else:
                raise ValueError(rule.family)
            execute_pos = signal_pos + 1
            if hit and execute_pos < exit_pos:
                item["exit_date_test"] = pd.Timestamp(dates[execute_pos])
                item["exit_close_test"] = float(closes[execute_pos])
                item["exit_reason_test"] = rule.key
                item["open_mark_test"] = False
                item["path_triggered"] = True
                item["path_signal_date"] = pd.Timestamp(dates[signal_pos])
                triggered = True
                break
        item["gross_return_test"] = float(item["exit_close_test"]) / entry_price - 1.0
        item["path_triggered"] = bool(triggered)
        rows.append(item)
    return pd.DataFrame(rows)


def profile(frame: pd.DataFrame, years: list[int]) -> dict[int, dict]:
    _, detail, _ = core.simulate_account(frame)
    detail_year = pd.to_datetime(detail["entry_date"]).dt.year
    source_year = pd.to_datetime(frame["entry_date"]).dt.year
    result = {}
    for year in years:
        local = detail[detail_year.lt(year)].copy()
        source = frame[source_year.lt(year)]
        if local.empty:
            result[year] = {
                "trades": 0,
                "final_value": core.INITIAL_CAPITAL,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
                "triggers": 0,
            }
            continue
        returns = local["account_return_test"].astype(float)
        equity = core.INITIAL_CAPITAL * (1.0 + returns).cumprod()
        closed = local[~local["open_mark_test"].astype(bool)]
        result[year] = {
            "trades": int(len(local)),
            "final_value": float(local.iloc[-1]["cash_after_test"]),
            "win_rate": float((closed["account_return_test"] > 0).mean()),
            "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
            "triggers": int(source.get("path_triggered", pd.Series(False, index=source.index)).astype(bool).sum()),
        }
    return result


def choose(
    profiles: dict[str, dict[int, dict]], families: dict[str, str], year: int
) -> tuple[str, dict]:
    base = profiles["baseline"][year]
    candidates = []
    for name, values in profiles.items():
        if name == "baseline":
            continue
        current = values[year]
        ratio = current["final_value"] / base["final_value"]
        win_delta = current["win_rate"] - base["win_rate"]
        dd_delta = current["max_drawdown"] - base["max_drawdown"]
        passes = bool(
            current["trades"] >= MIN_TRAIN_TRADES
            and current["triggers"] >= 2
            and ratio > 1.0
            and win_delta > 0
            and dd_delta >= -0.005
        )
        candidates.append(
            {
                "year": year,
                "rule": name,
                "family": families[name],
                "passes": passes,
                "ratio": ratio,
                "win_delta": win_delta,
                "dd_delta": dd_delta,
                "triggers": current["triggers"],
                "score": np.log(max(ratio, 1e-12)) + 1.0 * win_delta + 0.2 * min(dd_delta, 0.2),
            }
        )
    passing = [row for row in candidates if row["passes"]]
    family_counts = pd.Series([row["family"] for row in passing]).value_counts().to_dict()
    stable = [row for row in passing if family_counts.get(row["family"], 0) >= 3]
    if not stable:
        return "baseline", {"year": year, "rule": "baseline", "family": "baseline"}
    winner = max(stable, key=lambda row: row["score"])
    return str(winner["rule"]), winner


def validate_line(spec: core.LineSpec, path: Path, candidates: list[PathRule]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"symbol": str})
    frame["symbol"] = frame["symbol"].str.zfill(6)
    for column in ("entry_date", "exit_date_test", "control_signal_date"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    prices = core.load_prices(spec.project, set(frame["symbol"]))
    frames = {"baseline": frame}
    families = {"baseline": "baseline"}
    for rule in candidates:
        frames[rule.key] = apply_rule(frame, prices, rule)
        families[rule.key] = rule.family
    years = sorted(pd.to_datetime(frame["entry_date"]).dt.year.unique())
    profiles = {name: profile(candidate, years) for name, candidate in frames.items()}
    parts = []
    selections = []
    for year in years:
        rule, selection = choose(profiles, families, int(year))
        local = frames[rule][pd.to_datetime(frames[rule]["entry_date"]).dt.year.eq(year)].copy()
        local["walkforward_path_rule"] = rule
        parts.append(local)
        selections.append(selection)
    combined = pd.concat(parts, ignore_index=True).sort_values(["entry_date", "symbol"])
    baseline = {period: search.subset_stats(frame, period) for period in ("full", "dev", "holdout")}
    winner = {period: search.subset_stats(combined, period) for period in ("full", "dev", "holdout")}
    return {
        "line": spec.key,
        "candidate_rules": len(candidates),
        "baseline": baseline,
        "winner": winner,
        "full_ratio": winner["full"]["final_value"] / baseline["full"]["final_value"],
        "full_win_delta": winner["full"]["win_rate"] - baseline["full"]["win_rate"],
        "holdout_ratio": winner["holdout"]["final_value"] / baseline["holdout"]["final_value"],
        "holdout_win_delta": winner["holdout"]["win_rate"] - baseline["holdout"]["win_rate"],
        "selected_years": int(sum(row["rule"] != "baseline" for row in selections)),
    }, combined, pd.DataFrame(selections)


def main() -> None:
    core.configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        line: ROOT / "fit" / "rolling_upgrades" / f"{line.lower()}_trades.csv"
        for line in ("C", "S", "D")
    }
    specs = {spec.key: spec for spec in core.SPECS}
    candidates = rules()
    results = []
    for line in ("C", "S", "D"):
        payload, trades, selections = validate_line(specs[line], paths[line], candidates)
        results.append(payload)
        trades.to_csv(OUT / f"{line.lower()}_trades.csv", index=False, encoding="utf-8-sig")
        selections.to_csv(OUT / f"{line.lower()}_selections.csv", index=False, encoding="utf-8-sig")
    summary = {
        "method": "annual expanding-window path-quality exit selector",
        "execution": "close signal, next trading-day close exit",
        "results": results,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
