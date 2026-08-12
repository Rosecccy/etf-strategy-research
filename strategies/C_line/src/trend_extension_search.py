from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import controlled_upgrade_search as search
import take_profit_10_test as core
import walkforward_overlay_validation as wf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "trend_extension"


@dataclass(frozen=True)
class ExtendRule:
    ret20: float
    trail: float
    max_days: int

    @property
    def key(self) -> str:
        return f"extend:r{self.ret20:+.3f}:t{self.trail:.3f}:m{self.max_days}"


def indicators(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    close = frame["close"].astype(float)
    frame["ma20"] = close.rolling(20, min_periods=15).mean()
    frame["ma60"] = close.rolling(60, min_periods=40).mean()
    frame["ret20"] = close.pct_change(20)
    return frame


def apply_rule(
    frame: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    rule: ExtendRule,
) -> pd.DataFrame:
    result_rows = []
    ordered = frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)
    next_entries = list(pd.to_datetime(ordered["entry_date"]).shift(-1))
    for index, trade in ordered.iterrows():
        item = trade.to_dict()
        item["extend_rule"] = rule.key
        item["extend_triggered"] = False
        item["extend_signal_date"] = pd.NaT
        if bool(trade["open_mark_test"]):
            result_rows.append(item)
            continue
        data = indicators(prices[str(trade["symbol"])])
        dates = pd.DatetimeIndex(data["date"])
        base_exit = pd.Timestamp(trade["exit_date_test"])
        base_pos = int(dates.searchsorted(base_exit, side="left"))
        if base_pos >= len(data):
            result_rows.append(item)
            continue
        row = data.iloc[base_pos]
        strong = bool(
            pd.notna(row["ma20"])
            and pd.notna(row["ma60"])
            and pd.notna(row["ret20"])
            and float(row["close"]) > float(row["ma20"]) > float(row["ma60"])
            and float(row["ret20"]) >= rule.ret20
        )
        if not strong:
            result_rows.append(item)
            continue
        latest_pos = min(base_pos + rule.max_days, len(data) - 1)
        next_entry = next_entries[index]
        if pd.notna(next_entry):
            rotate_pos = int(dates.searchsorted(pd.Timestamp(next_entry), side="left"))
            latest_pos = min(latest_pos, rotate_pos)
        if latest_pos <= base_pos:
            result_rows.append(item)
            continue
        peak = float(data.iloc[base_pos]["close"])
        chosen_pos = latest_pos
        signal_date = pd.Timestamp(data.iloc[max(latest_pos - 1, base_pos)]["date"])
        for signal_pos in range(base_pos + 1, latest_pos):
            current = float(data.iloc[signal_pos]["close"])
            peak = max(peak, current)
            ma20 = float(data.iloc[signal_pos]["ma20"])
            break_trend = current < ma20 or current / peak - 1.0 <= -rule.trail
            if break_trend:
                chosen_pos = signal_pos + 1
                signal_date = pd.Timestamp(data.iloc[signal_pos]["date"])
                break
        if chosen_pos > base_pos:
            item["exit_date_test"] = pd.Timestamp(data.iloc[chosen_pos]["date"])
            item["exit_close_test"] = float(data.iloc[chosen_pos]["close"])
            item["exit_reason_test"] = rule.key
            item["gross_return_test"] = float(item["exit_close_test"]) / float(item["entry_close"]) - 1.0
            item["extend_triggered"] = True
            item["extend_signal_date"] = signal_date
        result_rows.append(item)
    return pd.DataFrame(result_rows)


def choose(
    profiles: dict[str, dict[int, dict]], families: dict[str, ExtendRule], year: int
) -> tuple[str, dict]:
    base = profiles["baseline"][year]
    rows = []
    for name, profile in profiles.items():
        if name == "baseline":
            continue
        current = profile[year]
        ratio = current["final_value"] / base["final_value"]
        win_delta = current["win_rate"] - base["win_rate"]
        dd_delta = current["max_drawdown"] - base["max_drawdown"]
        triggered = current.get("trigger_count", 0)
        passes = bool(
            current["trades"] >= 12
            and triggered >= 2
            and ratio > 1.0
            and win_delta >= 0
            and dd_delta >= -0.005
        )
        rows.append(
            {
                "year": year,
                "rule": name,
                "ret20": families[name].ret20,
                "trail": families[name].trail,
                "max_days": families[name].max_days,
                "ratio": ratio,
                "win_delta": win_delta,
                "passes": passes,
                "score": np.log(max(ratio, 1e-12)) + 0.8 * win_delta + 0.2 * min(dd_delta, 0.2),
            }
        )
    passing = [row for row in rows if row["passes"]]
    if len(passing) < 3:
        return "baseline", {"year": year, "rule": "baseline"}
    winner = max(passing, key=lambda row: row["score"])
    return str(winner["rule"]), winner


def validate_line(spec: core.LineSpec, path: Path, candidate_rules: list[ExtendRule]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"symbol": str})
    frame["symbol"] = frame["symbol"].str.zfill(6)
    for column in ("entry_date", "exit_date_test", "control_signal_date"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    prices = core.load_prices(spec.project, set(frame["symbol"]))
    frames = {"baseline": frame}
    rule_map = {}
    for rule in candidate_rules:
        adjusted = apply_rule(frame, prices, rule)
        adjusted["control_triggered"] = adjusted["extend_triggered"]
        frames[rule.key] = adjusted
        rule_map[rule.key] = rule
    years = sorted(pd.to_datetime(frame["entry_date"]).dt.year.unique())
    profiles = {name: wf.training_profile(candidate, years) for name, candidate in frames.items()}
    parts = []
    selections = []
    for year in years:
        rule, selection = choose(profiles, rule_map, int(year))
        local = frames[rule][pd.to_datetime(frames[rule]["entry_date"]).dt.year.eq(year)].copy()
        local["walkforward_extend_rule"] = rule
        parts.append(local)
        selections.append(selection)
    combined = pd.concat(parts, ignore_index=True).sort_values(["entry_date", "symbol"])
    baseline = {period: search.subset_stats(frame, period) for period in ("full", "dev", "holdout")}
    winner = {period: search.subset_stats(combined, period) for period in ("full", "dev", "holdout")}
    return {
        "line": spec.key,
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
    candidate_rules = [
        ExtendRule(ret20, trail, max_days)
        for ret20 in (0.00, 0.03, 0.05, 0.08, 0.10)
        for trail in (0.03, 0.05, 0.07, 0.10)
        for max_days in (5, 10, 15, 20, 30, 40)
    ]
    results = []
    for line in ("C", "S", "D"):
        payload, trades, selections = validate_line(specs[line], paths[line], candidate_rules)
        results.append(payload)
        trades.to_csv(OUT / f"{line.lower()}_trades.csv", index=False, encoding="utf-8-sig")
        selections.to_csv(OUT / f"{line.lower()}_selections.csv", index=False, encoding="utf-8-sig")
    summary = {
        "method": "annual expanding-window strong-trend exit extension",
        "execution": "decision near close; exit on next close after trend break, capped by next strategy entry",
        "results": results,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
