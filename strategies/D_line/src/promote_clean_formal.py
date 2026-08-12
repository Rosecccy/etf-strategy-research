from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import daily_panic_rolling_test as base
import daily_panic_stale_exit_test as stale
from control_panic_age_test import portfolio_metrics


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "out" / "daily_panic_stale_exit_test"
FORMAL = ROOT / "out" / "formal_clean"
CFG = ROOT / "cfg" / "daily_panic.json"
FORMAL_YEARS = tuple(range(2019, 2027))
HOLDOUT_YEARS = (2024, 2025, 2026)


def annual_table(log: pd.DataFrame) -> pd.DataFrame:
    if log.empty:
        return pd.DataFrame()
    local = log.copy()
    local["entry_date"] = pd.to_datetime(local["entry_date"], errors="coerce")
    local["net_return"] = pd.to_numeric(local["net_return"], errors="coerce")
    rows = []
    for year, group in local.groupby(local["entry_date"].dt.year, sort=True):
        closed = group[group["status_portfolio"].eq("closed") & group["net_return"].notna()]
        values = closed["net_return"].astype(float)
        rows.append(
            {
                "year": int(year),
                "accepted_trades": int(len(group)),
                "closed_trades": int(len(values)),
                "win_rate": float((values > 0).mean()) if len(values) else np.nan,
                "compounded_net_return": float(np.prod(1.0 + values) - 1.0) if len(values) else 0.0,
                "net_pnl": float(pd.to_numeric(closed.get("net_pnl"), errors="coerce").sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    quality = json.loads(
        (ROOT.parent / "C" / "raw" / "quality.json").read_text(encoding="utf-8")
    )
    if not bool(quality.get("passed")):
        raise RuntimeError("C canonical quality gate failed; D promotion is blocked.")

    stale_summary = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    if not bool(stale_summary.get("promoted")):
        raise RuntimeError("The clean D stale-exit candidate did not pass promotion gates.")
    chosen_window = stale_summary["development_selected_window"]
    window = None if str(chosen_window) == "all" else int(chosen_window)

    adjusted, raw, pool = base.prepare()
    formal = stale.build_formal(adjusted, raw, pool)
    choices = pd.read_csv(SOURCE / "winner_choices.csv", encoding="utf-8-sig")
    rules = {rule.key: rule for rule in stale.RULES}
    parts = []
    for _, choice in choices.iterrows():
        year = int(choice["year"])
        if year not in FORMAL_YEARS:
            continue
        key = str(choice["rule"])
        current = formal[formal["test_year"].eq(year)].copy()
        if key != "BASE_EXIT":
            current = stale.adjust_exits(current, raw, rules[key])
        current["selected_stale_exit"] = key
        current["selector_window"] = "all" if window is None else window
        parts.append(current)
    candidates = pd.concat(parts, ignore_index=True, sort=False)
    full_log, full_stat = portfolio_metrics(candidates, raw, pool)
    holdout_log, holdout_stat = portfolio_metrics(
        candidates[candidates["test_year"].isin(HOLDOUT_YEARS)].copy(), raw, pool
    )

    approved = {str(value).zfill(6) for value in quality["approved_symbols"]}
    used = set(full_log["symbol"].astype(str).str.zfill(6))
    unknown = sorted(used - approved)
    if unknown:
        raise RuntimeError(f"Unapproved ETF in formal D history: {unknown}")

    FORMAL.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(FORMAL / "candidate_trades.csv", index=False, encoding="utf-8-sig")
    full_log.to_csv(FORMAL / "account_trades.csv", index=False, encoding="utf-8-sig")
    holdout_log.to_csv(FORMAL / "holdout_trades.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(FORMAL / "selected_by_year.csv", index=False, encoding="utf-8-sig")
    annual_table(full_log).to_csv(FORMAL / "annual.csv", index=False, encoding="utf-8-sig")
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "formal_clean",
        "data_quality_passed": True,
        "approved_universe": len(approved),
        "selection": "Annual past-only panic thresholds and stale-exit rule; next-close execution.",
        "development_years": "2019-2023",
        "holdout_years": "2024-2026",
        "selected_window": chosen_window,
        "all": full_stat,
        "holdout": holdout_stat,
        "source_audit": "out/daily_panic_stale_exit_test/summary.json",
    }
    (FORMAL / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    cfg = json.loads(CFG.read_text(encoding="utf-8-sig"))
    cfg["version"] = "D3-clean-20260801"
    cfg["status"] = "formal_clean"
    cfg["universe"] = "C canonical 30-ETF clean whitelist"
    cfg["audit"] = {
        "summary": "out/formal_clean/summary.json",
        "annual_choices": "out/formal_clean/selected_by_year.csv",
        "formal_trades": "out/formal_clean/account_trades.csv",
        "data_quality": "C/raw/quality.json",
    }
    CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
