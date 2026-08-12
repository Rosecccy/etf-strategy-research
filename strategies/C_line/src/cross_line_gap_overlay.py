from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import controlled_upgrade_search as stats_mod
import take_profit_10_test as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "cross_line_gap"
MIN_TRAIN_BASE_TRADES = 12
MIN_TRAIN_OVERLAY_TRADES = 2

SOURCES = {
    "C": ROOT / "fit" / "rolling_upgrades" / "c_trades.csv",
    "S": ROOT / "fit" / "trend_extension" / "s_trades.csv",
    "D": ROOT / "fit" / "novel_path_exits" / "d_trades.csv",
}
PROJECTS = {"C": ROOT, "S": ROOT.parent / "S", "D": ROOT}


def load_frame(line: str) -> pd.DataFrame:
    frame = pd.read_csv(SOURCES[line], encoding="utf-8-sig", dtype={"symbol": str})
    frame["symbol"] = frame["symbol"].str.zfill(6)
    for column in (
        "entry_date",
        "exit_date_test",
        "control_signal_date",
        "extend_signal_date",
        "path_signal_date",
    ):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["origin_line"] = line
    frame["is_gap_overlay"] = False
    return frame.sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def load_all_prices(frames: dict[str, pd.DataFrame]) -> dict[tuple[str, str], pd.DataFrame]:
    result = {}
    for line, frame in frames.items():
        prices = core.load_prices(PROJECTS[line], set(frame["symbol"]))
        for symbol, data in prices.items():
            result[(line, symbol)] = data
    return result


def close_on(prices: dict[tuple[str, str], pd.DataFrame], line: str, symbol: str, date: pd.Timestamp) -> float:
    data = prices[(line, symbol)]
    row = data[data["date"].eq(date)]
    if row.empty:
        row = data[data["date"].lt(date)].tail(1)
    if row.empty:
        raise ValueError(f"no price for {line} {symbol} at {date.date()}")
    return float(row.iloc[0]["close"])


def overlay_options(target: str) -> list[tuple[str, ...]]:
    donors = [line for line in ("C", "S", "D") if line != target]
    result: list[tuple[str, ...]] = []
    for size in range(1, len(donors) + 1):
        result.extend(itertools.permutations(donors, size))
    return result


def build_schedule(
    target_line: str,
    target: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    priority: tuple[str, ...],
    prices: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    base = target.sort_values(["entry_date", "symbol"]).reset_index(drop=True)
    rows = []
    priority_rank = {line: rank for rank, line in enumerate(priority)}
    donor_pool = pd.concat([frames[line].copy() for line in priority], ignore_index=True)
    donor_pool = donor_pool.sort_values(["entry_date", "origin_line", "symbol"]).reset_index(drop=True)

    for index, trade in base.iterrows():
        item = trade.to_dict()
        item["overlay_priority"] = ">".join(priority)
        rows.append(item)
        gap_start = pd.Timestamp(trade["exit_date_test"])
        if index + 1 < len(base):
            gap_end = pd.Timestamp(base.iloc[index + 1]["entry_date"])
        else:
            gap_end = core.AS_OF + pd.Timedelta(days=1)
        cursor = gap_start
        while cursor < gap_end:
            eligible = donor_pool[
                donor_pool["entry_date"].gt(cursor)
                & donor_pool["entry_date"].lt(gap_end)
            ].copy()
            if eligible.empty:
                break
            first_date = eligible["entry_date"].min()
            same_day = eligible[eligible["entry_date"].eq(first_date)].copy()
            same_day["priority_rank"] = same_day["origin_line"].map(priority_rank)
            donor = same_day.sort_values(["priority_rank", "symbol"]).iloc[0]
            overlay = donor.to_dict()
            overlay["is_gap_overlay"] = True
            overlay["overlay_target_line"] = target_line
            overlay["overlay_priority"] = ">".join(priority)
            overlay["overlay_original_exit_date"] = pd.Timestamp(donor["exit_date_test"])
            exit_date = pd.Timestamp(donor["exit_date_test"])
            if exit_date >= gap_end:
                exit_date = gap_end
                overlay["exit_date_test"] = exit_date
                overlay["exit_close_test"] = close_on(
                    prices,
                    str(donor["origin_line"]),
                    str(donor["symbol"]),
                    exit_date,
                )
                overlay["exit_reason_test"] = "gap_preempted_by_base_signal"
                overlay["open_mark_test"] = False
                overlay["gross_return_test"] = (
                    float(overlay["exit_close_test"]) / float(overlay["entry_close"]) - 1.0
                )
            rows.append(overlay)
            cursor = pd.Timestamp(overlay["exit_date_test"])
            if cursor >= gap_end:
                break
    return pd.DataFrame(rows).sort_values(["entry_date", "is_gap_overlay", "symbol"]).reset_index(drop=True)


def choose_option(
    target: pd.DataFrame,
    candidates: dict[tuple[str, ...], pd.DataFrame],
    year: int,
) -> tuple[tuple[str, ...] | None, dict]:
    target_year = pd.to_datetime(target["entry_date"]).dt.year
    train_base = target[target_year.lt(year)].copy()
    if len(train_base) < MIN_TRAIN_BASE_TRADES:
        return None, {"test_year": year, "priority": "baseline", "reason": "insufficient_history"}
    base = core.simulate_account(train_base)[0]
    passing = []
    for priority, schedule in candidates.items():
        train = schedule[pd.to_datetime(schedule["entry_date"]).dt.year.lt(year)].copy()
        current = core.simulate_account(train)[0]
        overlay_count = int(train["is_gap_overlay"].fillna(False).sum())
        ratio = current["final_value"] / base["final_value"]
        win_delta = current["win_rate"] - base["win_rate"]
        dd_delta = current["max_drawdown"] - base["max_drawdown"]
        passes = bool(
            overlay_count >= MIN_TRAIN_OVERLAY_TRADES
            and ratio > 1.005
            and win_delta >= -1e-12
            and dd_delta >= -0.01
        )
        score = np.log(max(ratio, 1e-12)) + 0.8 * win_delta + 0.2 * min(dd_delta, 0.10)
        if passes:
            passing.append(
                {
                    "test_year": year,
                    "priority": ">".join(priority),
                    "priority_tuple": priority,
                    "reason": "past_only_improvement",
                    "train_base_trades": len(train_base),
                    "train_overlay_trades": overlay_count,
                    "train_ratio": ratio,
                    "train_win_delta": win_delta,
                    "train_dd_delta": dd_delta,
                    "score": score,
                }
            )
    if not passing:
        return None, {"test_year": year, "priority": "baseline", "reason": "no_past_only_improvement"}
    winner = max(passing, key=lambda row: row["score"])
    priority = winner.pop("priority_tuple")
    return priority, winner


def evaluate_line(
    line: str,
    frames: dict[str, pd.DataFrame],
    prices: dict[tuple[str, str], pd.DataFrame],
) -> dict:
    target = frames[line]
    options = overlay_options(line)
    candidates = {
        priority: build_schedule(line, target, frames, priority, prices) for priority in options
    }
    fixed_rows = []
    fixed_base = {period: stats_mod.subset_stats(target, period) for period in ("full", "dev", "holdout")}
    for priority, schedule in candidates.items():
        current = {period: stats_mod.subset_stats(schedule, period) for period in ("full", "dev", "holdout")}
        fixed_rows.append(
            {
                "priority": ">".join(priority),
                "trades": len(schedule),
                "added": int(schedule["is_gap_overlay"].fillna(False).sum()),
                "full_ratio": current["full"]["final_value"] / fixed_base["full"]["final_value"],
                "full_win": current["full"]["win_rate"],
                "full_avg_annual": current["full"]["avg_annual_return"],
                "full_max_drawdown": current["full"]["max_drawdown"],
                "holdout_ratio": current["holdout"]["final_value"] / fixed_base["holdout"]["final_value"],
                "holdout_win": current["holdout"]["win_rate"],
                "dev_ratio": current["dev"]["final_value"] / fixed_base["dev"]["final_value"],
                "dev_win": current["dev"]["win_rate"],
                "dev_max_drawdown": current["dev"]["max_drawdown"],
            }
        )
        schedule.to_csv(
            OUT / f"{line.lower()}_fixed_{'_'.join(priority).lower()}_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )
    fixed_table = pd.DataFrame(fixed_rows).sort_values("full_ratio", ascending=False)
    fixed_table.to_csv(OUT / f"{line.lower()}_fixed_candidates.csv", index=False, encoding="utf-8-sig")
    years = sorted(pd.to_datetime(target["entry_date"]).dt.year.unique())
    parts = []
    selections = []
    for year in years:
        priority, record = choose_option(target, candidates, int(year))
        source = target if priority is None else candidates[priority]
        local = source[pd.to_datetime(source["entry_date"]).dt.year.eq(year)].copy()
        local["walkforward_gap_priority"] = "baseline" if priority is None else ">".join(priority)
        parts.append(local)
        selections.append(record)
    winner = pd.concat(parts, ignore_index=True).sort_values(["entry_date", "is_gap_overlay", "symbol"])
    base_stats = {period: stats_mod.subset_stats(target, period) for period in ("full", "holdout")}
    winner_stats = {period: stats_mod.subset_stats(winner, period) for period in ("full", "holdout")}
    start = int(pd.to_datetime(target["entry_date"]).dt.year.min())
    years_count = 2026 - start + 1
    payload = {
        "line": line,
        "baseline": base_stats,
        "winner": winner_stats,
        "baseline_triggers": len(target),
        "winner_triggers": len(winner),
        "added_triggers": int(winner["is_gap_overlay"].fillna(False).sum()),
        "trigger_ratio": len(winner) / len(target),
        "baseline_avg_triggers_per_year": len(target) / years_count,
        "winner_avg_triggers_per_year": len(winner) / years_count,
        "full_ratio": winner_stats["full"]["final_value"] / base_stats["full"]["final_value"],
        "holdout_ratio": winner_stats["holdout"]["final_value"] / base_stats["holdout"]["final_value"],
        "top_fixed_candidates": fixed_table.head(4).to_dict(orient="records"),
    }
    winner.to_csv(OUT / f"{line.lower()}_trades.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selections).to_csv(OUT / f"{line.lower()}_selections.csv", index=False, encoding="utf-8-sig")
    return payload


def main() -> None:
    core.configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {line: load_frame(line) for line in ("C", "S", "D")}
    prices = load_all_prices(frames)
    results = [evaluate_line(line, frames, prices) for line in ("C", "S", "D")]
    payload = {
        "method": "annual past-only cross-line signals used only during the target line's cash gaps",
        "constraint": "all original target trades remain; overlays may only add triggers",
        "results": results,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
