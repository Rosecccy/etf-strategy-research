from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import clean_cs_control_sweep as sweep


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "fit" / "clean_control_sweep"
OUT = ROOT / "fit" / "clean_upgrades"
DEV_YEARS = sweep.DEV_YEARS
HOLDOUT_YEARS = sweep.HOLDOUT_YEARS


def mixed_by_evidence(
    baseline: pd.DataFrame,
    extension: pd.DataFrame,
    choices: pd.DataFrame,
    floor: int,
) -> tuple[pd.DataFrame, list[int]]:
    active = set(
        choices.loc[
            pd.to_numeric(choices["train_trades"], errors="coerce").ge(floor)
            & choices["rule"].ne("BASE"),
            "year",
        ].astype(int)
    )
    mixed = pd.concat(
        [
            extension[extension["year"].isin(active)],
            baseline[~baseline["year"].isin(active)],
        ],
        ignore_index=True,
        sort=False,
    ).sort_values(["entry_date", "symbol"])
    return mixed, sorted(active)


def annual_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in range(int(frame["year"].min()), 2027):
        local = frame[frame["year"].eq(year)]
        values = pd.to_numeric(local["ret"], errors="coerce").dropna().astype(float)
        rows.append(
            {
                "year": year,
                "trades": int(len(values)),
                "win_rate": float((values > 0).mean()) if len(values) else np.nan,
                "gross_return": float(np.prod(1.0 + values) - 1.0) if len(values) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def evaluate_line(line: str, floors: tuple[int, ...]) -> dict:
    if line == "C":
        baseline_path = ROOT / "fit" / "formal_clean" / "final_trades.csv"
    else:
        baseline_path = ROOT.parent / "S" / "fit" / "selector" / "formal_execution_trades.csv"
    baseline = sweep.read_trades(baseline_path)
    extension = sweep.read_trades(SOURCE / f"{line.lower()}_extension_trades.csv")
    choices = pd.read_csv(SOURCE / f"{line.lower()}_extension_choices.csv", encoding="utf-8-sig")
    base_full = sweep.metrics(baseline)
    base_dev = sweep.metrics(baseline, DEV_YEARS)
    base_holdout = sweep.metrics(baseline, HOLDOUT_YEARS)
    rows = []
    generated = {}
    for floor in floors:
        trades, active = mixed_by_evidence(baseline, extension, choices, floor)
        generated[floor] = (trades, active)
        full = sweep.metrics(trades)
        dev = sweep.metrics(trades, DEV_YEARS)
        holdout = sweep.metrics(trades, HOLDOUT_YEARS)
        rows.append(
            {
                "evidence_floor": floor,
                "active_years": ",".join(map(str, active)),
                **{f"full_{k}": v for k, v in full.items()},
                **{f"dev_{k}": v for k, v in dev.items()},
                **{f"holdout_{k}": v for k, v in holdout.items()},
            }
        )
    table = pd.DataFrame(rows)
    eligible = table[
        table["dev_gross_final_1000"].gt(base_dev["gross_final_1000"])
        & table["dev_win_rate"].ge(base_dev["win_rate"])
        & table["dev_avg_annual_return"].gt(base_dev["avg_annual_return"])
        & table["dev_max_drawdown"].ge(base_dev["max_drawdown"] - 0.005)
    ].copy()
    if eligible.empty:
        chosen = floors[-1]
    else:
        eligible["selection_score"] = (
            np.log(eligible["dev_gross_final_1000"] / 1000.0)
            + 0.70 * eligible["dev_win_rate"]
            + 0.75 * eligible["dev_avg_annual_return"]
            + 0.80 * eligible["dev_max_drawdown"]
        )
        chosen = int(eligible.sort_values("selection_score", ascending=False).iloc[0]["evidence_floor"])
    winner, active = generated[chosen]
    full = sweep.metrics(winner)
    holdout = sweep.metrics(winner, HOLDOUT_YEARS)
    improved_neighbors = int(
        (
            table["full_net_final_10000"].gt(base_full["net_final_10000"])
            & table["full_win_rate"].ge(base_full["win_rate"])
            & table["full_avg_annual_return"].gt(base_full["avg_annual_return"])
            & table["holdout_gross_final_1000"].gt(base_holdout["gross_final_1000"])
            & table["holdout_win_rate"].ge(base_holdout["win_rate"])
            & table["holdout_avg_annual_return"].gt(base_holdout["avg_annual_return"])
        ).sum()
    )
    promoted = bool(
        full["net_final_10000"] > base_full["net_final_10000"]
        and full["win_rate"] >= base_full["win_rate"]
        and full["avg_annual_return"] > base_full["avg_annual_return"]
        and holdout["gross_final_1000"] > base_holdout["gross_final_1000"]
        and holdout["win_rate"] >= base_holdout["win_rate"]
        and holdout["avg_annual_return"] > base_holdout["avg_annual_return"]
        and holdout["max_drawdown"] >= base_holdout["max_drawdown"] - 0.005
        and improved_neighbors >= 3
    )
    prefix = line.lower()
    table.to_csv(OUT / f"{prefix}_evidence.csv", index=False, encoding="utf-8-sig")
    winner.to_csv(OUT / f"{prefix}_trades.csv", index=False, encoding="utf-8-sig")
    annual_table(winner).to_csv(OUT / f"{prefix}_annual.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(OUT / f"{prefix}_rules.csv", index=False, encoding="utf-8-sig")
    return {
        "line": line,
        "method": "annual past-only strong-trend exit extension with historical evidence gate",
        "selected_evidence_floor_on_2019_2023": chosen,
        "active_years": active,
        "baseline_full": base_full,
        "winner_full": full,
        "baseline_holdout": base_holdout,
        "winner_holdout": holdout,
        "stable_evidence_floors_improved": improved_neighbors,
        "promoted": promoted,
    }


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    OUT.mkdir(parents=True, exist_ok=True)
    c_quality = json.loads((ROOT / "raw" / "quality.json").read_text(encoding="utf-8"))
    s_quality = json.loads((ROOT.parent / "S" / "raw" / "quality.json").read_text(encoding="utf-8"))
    if not bool(c_quality.get("passed")) or not bool(s_quality.get("passed")):
        raise RuntimeError("Clean-data quality gate failed.")
    results = [
        evaluate_line("C", (15, 20, 25, 30, 32, 35, 40, 45)),
        evaluate_line("S", (5, 8, 10, 12, 15, 18, 20, 22, 25)),
    ]
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "quality_gate": "C 30/30 and S 61/62 approved; rejected data excluded",
        "selection": "evidence floor selected on 2019-2023; each test year uses only prior completed years; 2024-2026 is rolling OOS but no longer a blind holdout after repeated research",
        "execution": "state after close, next-close execution, 0.03% commission with CNY 5 minimum, 100-share lots",
        "results": results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
