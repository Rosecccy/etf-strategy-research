from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

import controlled_upgrade_search as search
import take_profit_10_test as core
import walkforward_overlay_validation as wf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "walkforward_policy_sweep"


def parse_value(rule: str, key: str) -> float | None:
    match = re.search(rf"(?:^|:){key}([+-]\d+\.\d+)", rule)
    return float(match.group(1)) if match else None


def parse_hold(rule: str) -> int | None:
    match = re.search(r":h(\d+)$", rule)
    return int(match.group(1)) if match else None


def allowed_frames(
    line: str,
    frames: dict[str, pd.DataFrame],
    value: float,
) -> dict[str, pd.DataFrame]:
    result = {"baseline": frames["baseline"]}
    for name, frame in frames.items():
        if name == "baseline":
            continue
        if line == "S":
            hold = parse_hold(name)
            allowed = hold is not None and hold >= int(value)
        elif line == "D":
            target = parse_value(name, "a")
            allowed = target is not None and target <= value + 1e-12
        else:
            allowed = True
        if allowed:
            result[name] = frame
    return result


def run_policy(
    line: str,
    value: float,
    frames: dict[str, pd.DataFrame],
    profiles: dict[str, dict[int, dict]],
    years: list[int],
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    allowed = allowed_frames(line, frames, value)
    local_profiles = {name: profiles[name] for name in allowed}
    selections = []
    parts = []
    for year in years:
        rule, selection, _ = wf.choose_rule(local_profiles, year)
        part = frames[rule][pd.to_datetime(frames[rule]["entry_date"]).dt.year.eq(year)].copy()
        part["walkforward_rule"] = rule
        part["walkforward_test_year"] = year
        parts.append(part)
        selections.append(selection)
    combined = pd.concat(parts, ignore_index=True).sort_values(["entry_date", "symbol"])
    stats = {
        "full": search.subset_stats(combined, "full"),
        "dev": search.subset_stats(combined, "dev"),
        "holdout": search.subset_stats(combined, "holdout"),
    }
    payload = {
        "line": line,
        "policy_value": value,
        "allowed_rules": len(allowed),
        "selected_nonbaseline_years": int(sum(row["rule"] != "baseline" for row in selections)),
        **{
            f"{period}_{metric}": values[metric]
            for period, values in stats.items()
            for metric in ("final_value", "win_rate", "max_drawdown", "avg_annual_return")
        },
    }
    return payload, combined, pd.DataFrame(selections)


def evaluate_line(spec: core.LineSpec, values: list[float]) -> dict:
    trades = core.load_trades(spec)
    prices = core.load_prices(spec.project, set(trades["symbol"]))
    frames = wf.candidate_frames(spec, trades, prices)
    years = sorted(pd.to_datetime(trades["entry_date"]).dt.year.unique())
    profiles = {name: wf.training_profile(frame, years) for name, frame in frames.items()}
    baseline = frames["baseline"]
    baseline_stats = {
        "full": search.subset_stats(baseline, "full"),
        "dev": search.subset_stats(baseline, "dev"),
        "holdout": search.subset_stats(baseline, "holdout"),
    }
    rows = []
    generated = {}
    selected = {}
    for value in values:
        payload, combined, selections = run_policy(spec.key, value, frames, profiles, years)
        rows.append(payload)
        generated[value] = combined
        selected[value] = selections
    table = pd.DataFrame(rows)
    table["dev_ratio"] = table["dev_final_value"] / baseline_stats["dev"]["final_value"]
    table["holdout_ratio"] = table["holdout_final_value"] / baseline_stats["holdout"]["final_value"]
    table["dev_win_delta"] = table["dev_win_rate"] - baseline_stats["dev"]["win_rate"]
    table["holdout_win_delta"] = table["holdout_win_rate"] - baseline_stats["holdout"]["win_rate"]
    table["dev_eligible"] = (
        table["dev_ratio"].gt(1.0)
        & table["dev_win_delta"].ge(-1e-12)
        & table["dev_max_drawdown"].ge(baseline_stats["dev"]["max_drawdown"] - 0.005)
    )
    eligible = table[table["dev_eligible"]]
    winner = eligible.sort_values("dev_ratio", ascending=False).iloc[0] if len(eligible) else table.iloc[0]
    winner_value = float(winner["policy_value"])
    table.to_csv(OUT / f"{spec.key.lower()}_policies.csv", index=False, encoding="utf-8-sig")
    generated[winner_value].to_csv(
        OUT / f"{spec.key.lower()}_winner_trades.csv", index=False, encoding="utf-8-sig"
    )
    selected[winner_value].to_csv(
        OUT / f"{spec.key.lower()}_winner_selections.csv", index=False, encoding="utf-8-sig"
    )
    return {
        "line": spec.key,
        "selection_uses": "development period only (through 2023)",
        "winner_policy_value": winner_value,
        "winner_passes_holdout": bool(
            winner["holdout_ratio"] > 1.0 and winner["holdout_win_delta"] >= -1e-12
        ),
        "baseline": baseline_stats,
        "winner": winner.to_dict(),
    }


def main() -> None:
    core.configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    specs = {spec.key: spec for spec in core.SPECS}
    results = [
        evaluate_line(specs["S"], [25, 28, 30, 32, 35, 40]),
        evaluate_line(specs["D"], [0.16, 0.18, 0.20, 0.22, 0.24, 0.25]),
    ]
    summary = {
        "method": "one-variable annual walk-forward selector constraint sweep",
        "results": results,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
