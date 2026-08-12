from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import controlled_upgrade_search as search
import take_profit_10_test as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "source_exit"


def candidate_rules() -> list[search.Rule]:
    result: list[search.Rule] = []
    result += [
        search.Rule("trailing", arm, trail, hold)
        for arm in (0.03, 0.05, 0.10, 0.15, 0.25)
        for trail in (0.02, 0.05, 0.10)
        for hold in (5, 10, 20)
        if trail < arm + 0.03
    ]
    result += [
        search.Rule("profit_floor", arm, floor, hold)
        for arm in (0.05, 0.10, 0.15, 0.20)
        for floor in (0.0, 0.02, 0.05, 0.08)
        for hold in (5, 10, 20)
        if floor < arm
    ]
    result += [
        search.Rule("loss_cut", threshold, 0.0, hold)
        for threshold in (-0.02, -0.03, -0.05, -0.07, -0.10)
        for hold in (5, 10, 20)
    ]
    result += [
        search.Rule("ma20_weak", threshold, 0.0, hold)
        for threshold in (-0.05, -0.03, -0.02, -0.01, 0.0)
        for hold in (10, 20)
    ]
    return result


def scopes(frame: pd.DataFrame) -> dict[str, pd.Series]:
    source = frame["source"].fillna("").astype(str)
    fallback = source.ne("\u4e3b\u7b56\u7565")
    result = {"fallback": fallback}
    for value in sorted(source[fallback].unique()):
        result[f"source={value}"] = source.eq(value)
    return result


def apply_scoped(
    frame: pd.DataFrame,
    adjusted: pd.DataFrame,
    mask: pd.Series,
    key: str,
) -> pd.DataFrame:
    result = frame.copy()
    columns = [
        "exit_date_test",
        "exit_close_test",
        "exit_reason_test",
        "open_mark_test",
        "gross_return_test",
        "control_triggered",
        "control_signal_date",
    ]
    for column in columns:
        if column in adjusted:
            result.loc[mask, column] = adjusted.loc[mask, column].to_numpy()
    result["source_exit_rule"] = key
    result["source_exit_applied"] = mask & result.get(
        "control_triggered", pd.Series(False, index=result.index)
    ).fillna(False).astype(bool)
    return result


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
            "triggers": int(source.get("source_exit_applied", pd.Series(False, index=source.index)).fillna(False).astype(bool).sum()),
        }
    return result


def choose(profiles: dict[str, dict[int, dict]], meta: dict[str, tuple[str, str]], year: int) -> tuple[str, dict]:
    base = profiles["baseline"][year]
    rows = []
    for key, values in profiles.items():
        if key == "baseline":
            continue
        current = values[year]
        ratio = current["final_value"] / base["final_value"]
        win_delta = current["win_rate"] - base["win_rate"]
        dd_delta = current["max_drawdown"] - base["max_drawdown"]
        scope, family = meta[key]
        passes = bool(
            current["trades"] >= 12
            and current["triggers"] >= 2
            and ratio > 1.0
            and win_delta >= 0.0
            and dd_delta >= -0.005
        )
        rows.append(
            {
                "year": year,
                "rule": key,
                "scope": scope,
                "family": family,
                "passes": passes,
                "ratio": ratio,
                "win_delta": win_delta,
                "dd_delta": dd_delta,
                "triggers": current["triggers"],
                "score": np.log(max(ratio, 1e-12)) + win_delta + 0.2 * min(dd_delta, 0.2),
            }
        )
    passing = [row for row in rows if row["passes"]]
    counts = pd.Series([(row["scope"], row["family"]) for row in passing]).value_counts().to_dict()
    stable = [row for row in passing if counts.get((row["scope"], row["family"]), 0) >= 3]
    if not stable:
        return "baseline", {"year": year, "rule": "baseline", "scope": "baseline", "family": "baseline"}
    winner = max(stable, key=lambda row: row["score"])
    return str(winner["rule"]), winner


def validate(line: str, rules: list[search.Rule]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    path = ROOT / "fit" / "rolling_upgrades" / f"{line.lower()}_trades.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"symbol": str})
    frame["symbol"] = frame["symbol"].str.zfill(6)
    frame["entry_date"] = pd.to_datetime(frame["entry_date"])
    for column in ("exit_date_test", "control_signal_date"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    project = ROOT if line == "C" else ROOT.parent / "S"
    prices = core.load_prices(project, set(frame["symbol"]))
    scope_map = scopes(frame)
    frames = {"baseline": frame}
    meta: dict[str, tuple[str, str]] = {}
    for rule in rules:
        adjusted = search.apply_rule(frame, prices, rule)
        for scope, mask in scope_map.items():
            key = f"{scope}|{rule.key}"
            frames[key] = apply_scoped(frame, adjusted, mask, key)
            meta[key] = (scope, rule.family)
    years = sorted(frame["entry_date"].dt.year.unique())
    profiles = {key: profile(candidate, years) for key, candidate in frames.items()}
    selections = []
    parts = []
    for year in years:
        key, audit = choose(profiles, meta, int(year))
        local = frames[key][frames[key]["entry_date"].dt.year.eq(year)].copy()
        local["walkforward_source_exit_rule"] = key
        parts.append(local)
        selections.append(audit)
    winner_frame = pd.concat(parts, ignore_index=True).sort_values(["entry_date", "symbol"])
    baseline = {period: search.subset_stats(frame, period) for period in ("full", "dev", "holdout")}
    winner = {period: search.subset_stats(winner_frame, period) for period in ("full", "dev", "holdout")}
    return {
        "line": line,
        "candidate_rules": len(frames) - 1,
        "baseline": baseline,
        "winner": winner,
        "full_ratio": winner["full"]["final_value"] / baseline["full"]["final_value"],
        "full_win_delta": winner["full"]["win_rate"] - baseline["full"]["win_rate"],
        "holdout_ratio": winner["holdout"]["final_value"] / baseline["holdout"]["final_value"],
        "holdout_win_delta": winner["holdout"]["win_rate"] - baseline["holdout"]["win_rate"],
        "selected_years": int(sum(row["rule"] != "baseline" for row in selections)),
    }, winner_frame, pd.DataFrame(selections)


def main() -> None:
    core.configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = candidate_rules()
    results = []
    for line in ("C", "S"):
        payload, trades, selections = validate(line, candidates)
        results.append(payload)
        trades.to_csv(OUT / f"{line.lower()}_trades.csv", index=False, encoding="utf-8-sig")
        selections.to_csv(OUT / f"{line.lower()}_selections.csv", index=False, encoding="utf-8-sig")
    summary = {
        "method": "annual expanding-window source-specific exit selector",
        "execution": "close signal and next trading-day close exit; main trades unchanged",
        "results": results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
