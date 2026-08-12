from __future__ import annotations

"""Annual causal selection of an interpretable eligibility filter.

Each January chooses one filter from only prior realised, already-confirmed
signals.  The selected filter is fixed for the full year.  Meta-parameters are
tuned on 2016-2019 and then locked for the 2020-2025 holdout.
"""

import json
from datetime import datetime

import pandas as pd

import rolling_extrema5_online as online
from ablate_and_ensemble_dividend_extrema5 import groups
from filter_extrema5_audit import (
    BASE_RATE, HIGH_CANDIDATE, HOLDOUT_YEARS, LOW_CANDIDATE, TUNE_YEARS,
    all_filters, candidate_predictions, metric, with_feature_rows,
)
from fit_dividend_extrema5 import OUT, PROJECT, SYMBOL, WINDOW, build_features


ALL_YEARS = range(2013, 2026)
META_TUNE_YEARS = range(2016, 2020)
META_HOLDOUT_YEARS = range(2020, 2026)


def choose_filter(history: pd.DataFrame, filters: list[tuple[str, callable]], memory: int, minimum: int) -> str:
    years = sorted(history["year"].unique())[-memory:] if memory else sorted(history["year"].unique())
    sample = history.loc[history["year"].isin(years)].copy()
    # At the start of a year, the last five prior-day labels are not confirmed.
    if len(sample) > WINDOW:
        sample = sample.iloc[:-WINDOW]
    choices = []
    for name, func in filters:
        stats = metric(sample, func(sample).fillna(False))
        if stats["signals"] < minimum:
            continue
        # Laplace smoothing prevents tiny samples from dominating the choice.
        posterior_precision = (stats["hits"] + 2) / (stats["signals"] + 6)
        choices.append((posterior_precision, stats["precision"], stats["recall"], name))
    if not choices:
        return "无额外过滤"
    return max(choices)[-1]


def apply_selector(frame: pd.DataFrame, filters: list[tuple[str, callable]], memory: int, minimum: int, years: range) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = dict(filters)
    decisions = []
    selected = []
    for year in years:
        history = frame.loc[frame["year"] < year]
        name = choose_filter(history, filters, memory, minimum)
        test = frame.loc[frame["year"] == year].copy()
        allowed = mapping[name](test).fillna(False)
        stats = metric(test, allowed)
        test["eligible"] = allowed.astype(int)
        test["selected_filter"] = name
        selected.append(test)
        decisions.append({"year": year, "selected_filter": name, **stats})
    return pd.concat(selected, ignore_index=True), pd.DataFrame(decisions)


def aggregate(decisions: pd.DataFrame) -> dict[str, float | int]:
    signals = int(decisions["signals"].sum())
    hits = int(decisions["hits"].sum())
    positives = int(decisions["recall"].mul(0).sum())  # Filled by caller from the event frame.
    return {
        "signals": signals,
        "hits": hits,
        "precision": hits / signals if signals else 0.0,
        "mean_year_precision": float(decisions["precision"].mean()),
        "min_year_precision": float(decisions["precision"].min()),
    }


def settings_grid() -> list[tuple[int, int]]:
    return [(memory, minimum) for memory in (0, 3, 5) for minimum in (20, 35, 50, 70)]


def run_target(target: str, frame: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    filters = all_filters(target)
    score_rows = []
    for memory, minimum in settings_grid():
        _, decisions = apply_selector(frame, filters, memory, minimum, META_TUNE_YEARS)
        result = aggregate(decisions)
        # A selector must issue at least four decisions per tuning year on average.
        score_rows.append({"memory": "all" if memory == 0 else memory, "minimum": minimum, **result})
    ranked = pd.DataFrame(score_rows)
    eligible = ranked.loc[ranked["signals"] >= len(META_TUNE_YEARS) * 4].copy()
    chosen = eligible.sort_values(
        ["precision", "min_year_precision", "mean_year_precision", "signals"], ascending=False
    ).iloc[0]
    memory = 0 if chosen["memory"] == "all" else int(chosen["memory"])
    minimum = int(chosen["minimum"])
    hold_signals, hold_decisions = apply_selector(frame, filters, memory, minimum, META_HOLDOUT_YEARS)
    selected = hold_signals.loc[(hold_signals["rank"] >= 1.0 - BASE_RATE) & (hold_signals["eligible"] == 1)]
    baseline = hold_signals.loc[hold_signals["rank"] >= 1.0 - BASE_RATE]
    result = {
        "target": target,
        "candidate": str(frame["candidate"].iloc[0]),
        "base_rate": BASE_RATE,
        "memory": chosen["memory"],
        "minimum_history_signals": minimum,
        "meta_tune_precision": float(chosen["precision"]),
        "meta_tune_signals": int(chosen["signals"]),
        "holdout_precision": float(selected["actual"].mean()) if len(selected) else 0.0,
        "holdout_recall": float(selected["actual"].sum() / hold_signals["actual"].sum()),
        "holdout_signals": int(len(selected)),
        "holdout_hits": int(selected["actual"].sum()),
        "holdout_baseline_precision": float(baseline["actual"].mean()),
        "holdout_precision_change": float(selected["actual"].mean() - baseline["actual"].mean()) if len(selected) else 0.0,
        "holdout_min_year_precision": float(hold_decisions["precision"].min()),
    }
    return result, ranked, hold_decisions


def main() -> None:
    raw = pd.read_csv(PROJECT / "C" / "raw" / "etf" / f"{SYMBOL}.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    features, labels = build_features(raw)
    online.FEATURE_GROUPS = groups(features)
    output_rows = []
    ranking_tables = []
    decision_tables = []
    for target, candidate in (("low", LOW_CANDIDATE), ("high", HIGH_CANDIDATE)):
        prediction = candidate_predictions(features, labels, target, candidate, ALL_YEARS)
        frame = with_feature_rows(prediction, features, labels)
        result, ranked, decisions = run_target(target, frame)
        output_rows.append(result)
        ranked.insert(0, "target", target)
        decisions.insert(0, "target", target)
        ranking_tables.append(ranked)
        decision_tables.append(decisions)
    result_frame = pd.DataFrame(output_rows)
    result_frame.to_csv(OUT / "adaptive_filter_selected.csv", index=False, encoding="utf-8-sig")
    pd.concat(ranking_tables, ignore_index=True).to_csv(OUT / "adaptive_filter_tuning_grid.csv", index=False, encoding="utf-8-sig")
    pd.concat(decision_tables, ignore_index=True).to_csv(OUT / "adaptive_filter_holdout_yearly.csv", index=False, encoding="utf-8-sig")
    payload = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "results": output_rows}
    (OUT / "adaptive_filter_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
