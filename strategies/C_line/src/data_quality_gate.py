from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
POOL = ROOT / "raw" / "pool.csv"
CANONICAL = ROOT / "raw" / "etf"
STAGING = WORKSPACE / "S" / "raw" / "etf"
S_POOL = WORKSPACE / "S" / "raw" / "pool.csv"
S_OUT = WORKSPACE / "S" / "raw" / "quality"
S_STATUS = WORKSPACE / "S" / "raw" / "quality.json"
S_QUARANTINE = WORKSPACE / "S" / "raw" / "quarantine"
REFERENCE = (
    WORKSPACE
    / "D"
    / "out"
    / "research_30_extrema_cleaned_v2"
    / "raw"
)
OUT = ROOT / "raw" / "quality"
STATUS = ROOT / "raw" / "quality.json"
HARD_JUMP = 0.205
REVIEW_JUMP = 0.115


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "amount"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    return frame.sort_values("date").reset_index(drop=True)


def audit_symbol(symbol: str, path: Path, source: str) -> tuple[dict, pd.DataFrame]:
    if not path.exists():
        return {
            "source": source,
            "symbol": symbol,
            "exists": False,
            "passed": False,
            "reason": "missing",
        }, pd.DataFrame()
    frame = load_frame(path)
    duplicate = int(frame["date"].duplicated().sum())
    invalid = int(
        (
            frame[["open", "high", "low", "close"]].le(0).any(axis=1)
            | frame["date"].isna()
        ).sum()
    )
    bad_ohlc = int(
        (
            frame["high"].lt(frame[["open", "close", "low"]].max(axis=1))
            | frame["low"].gt(frame[["open", "close", "high"]].min(axis=1))
        ).sum()
    )
    frame["return"] = frame["close"].pct_change()
    events = frame[frame["return"].abs().gt(REVIEW_JUMP)].copy()
    events["source"] = source
    events["symbol"] = symbol
    events["previous_close"] = frame["close"].shift(1).loc[events.index]
    events["hard_failure"] = events["return"].abs().gt(HARD_JUMP)
    events["event_type"] = np.where(
        events["hard_failure"], "impossible_jump", "limit_review"
    )
    hard = int(events["hard_failure"].sum())
    passed = duplicate == 0 and invalid == 0 and bad_ohlc == 0 and hard == 0
    return {
        "source": source,
        "symbol": symbol,
        "exists": True,
        "rows": int(len(frame)),
        "start": frame["date"].min().date().isoformat(),
        "end": frame["date"].max().date().isoformat(),
        "duplicate_dates": duplicate,
        "invalid_rows": invalid,
        "bad_ohlc": bad_ohlc,
        "review_jumps": int(len(events)),
        "hard_jumps": hard,
        "passed": passed,
        "reason": "ok" if passed else "structural_or_jump_failure",
    }, events[
        [
            "source",
            "symbol",
            "date",
            "previous_close",
            "close",
            "return",
            "event_type",
            "hard_failure",
        ]
    ]


def return_compare(symbol: str, left_path: Path, right_path: Path) -> dict:
    if not left_path.exists() or not right_path.exists():
        return {
            "symbol": symbol,
            "overlap_returns": 0,
            "return_correlation": np.nan,
            "max_abs_return_difference": np.nan,
            "passed": False,
        }
    stage = load_frame(left_path)[["date", "close"]].rename(
        columns={"close": "stage_close"}
    )
    ref = load_frame(right_path)[["date", "close"]].rename(
        columns={"close": "ref_close"}
    )
    merged = stage.merge(ref, on="date", how="inner").sort_values("date")
    merged["stage_ret"] = merged["stage_close"].pct_change()
    merged["ref_ret"] = merged["ref_close"].pct_change()
    clean = merged.dropna(subset=["stage_ret", "ref_ret"])
    corr = float(clean["stage_ret"].corr(clean["ref_ret"])) if len(clean) > 2 else np.nan
    max_diff = float((clean["stage_ret"] - clean["ref_ret"]).abs().max()) if len(clean) else np.nan
    passed = bool(len(clean) >= 50 and corr >= 0.985 and max_diff <= 0.06)
    return {
        "symbol": symbol,
        "overlap_returns": int(len(clean)),
        "return_correlation": corr,
        "max_abs_return_difference": max_diff,
        "passed": passed,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_directory(symbols: list[str], directory: Path, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    events = []
    for symbol in symbols:
        row, event = audit_symbol(symbol, directory / f"{symbol}.csv", source)
        rows.append(row)
        if not event.empty:
            events.append(event)
    return pd.DataFrame(rows), pd.concat(events, ignore_index=True) if events else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and synchronize the 30-ETF canonical data pool.")
    parser.add_argument("--sync", action="store_true", help="Replace C raw ETF files with audited S staging files.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(POOL, dtype={"symbol": str}, encoding="utf-8-sig")
    enabled = pool[pool["enabled"].astype(str).str.lower().isin(["true", "1", "yes"])]
    symbols = enabled["symbol"].astype(str).str.zfill(6).tolist()

    source_audits = []
    source_events = []
    audit_maps: dict[str, dict[str, dict]] = {}
    for label, directory in (
        ("C_existing", CANONICAL),
        ("S_staging", STAGING),
        ("D_reference", REFERENCE),
    ):
        audit, events = audit_directory(symbols, directory, label)
        source_audits.append(audit)
        if not events.empty:
            source_events.append(events)
        audit_maps[label] = audit.set_index("symbol").to_dict("index")

    plan_rows = []
    selected_paths: dict[str, Path] = {}
    for symbol in symbols:
        choice = None
        for label, directory in (
            ("C_existing", CANONICAL),
            ("S_staging", STAGING),
            ("D_reference", REFERENCE),
        ):
            if bool(audit_maps[label][symbol]["passed"]):
                choice = (label, directory / f"{symbol}.csv")
                break
        if choice is None:
            plan_rows.append({"symbol": symbol, "selected_source": "NONE", "passed": False})
            continue
        label, path = choice
        selected_paths[symbol] = path
        plan_rows.append({"symbol": symbol, "selected_source": label, "selected_path": str(path), "passed": True})
    source_plan = pd.DataFrame(plan_rows)
    comparison_rows = []
    for symbol, selected in selected_paths.items():
        reference = REFERENCE / f"{symbol}.csv"
        if selected.resolve() == reference.resolve():
            reference = STAGING / f"{symbol}.csv"
        item = return_compare(symbol, selected, reference)
        item["selected_path"] = str(selected)
        item["reference_path"] = str(reference)
        comparison_rows.append(item)
    comparison = pd.DataFrame(comparison_rows)
    sources_ok = bool(source_plan["passed"].all())
    compare_ok = bool(comparison["passed"].all())
    synced = False
    if args.sync and sources_ok and compare_ok:
        CANONICAL.mkdir(parents=True, exist_ok=True)
        for symbol, selected in selected_paths.items():
            target = CANONICAL / f"{symbol}.csv"
            if selected.resolve() != target.resolve():
                shutil.copy2(selected, target)
        synced = True

    canonical_audit, canonical_events = audit_directory(
        symbols, CANONICAL, "C_canonical"
    )
    canonical_ok = bool(canonical_audit["passed"].all())
    approved = canonical_audit.loc[canonical_audit["passed"], "symbol"].tolist()
    manifest = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "path": str(CANONICAL / f"{symbol}.csv"),
                "sha256": sha256(CANONICAL / f"{symbol}.csv"),
            }
            for symbol in approved
        ]
    )
    all_events = pd.concat(
        [item for item in (*source_events, canonical_events) if not item.empty],
        ignore_index=True,
    ) if source_events or not canonical_events.empty else pd.DataFrame()
    overall = bool(
        len(symbols) == 30
        and sources_ok
        and compare_ok
        and canonical_ok
        and len(approved) == len(symbols)
    )
    status = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "passed": overall,
        "canonical_source": str(CANONICAL),
        "staging_source": str(STAGING),
        "independent_reference": str(REFERENCE),
        "pool_symbols": len(symbols),
        "approved_symbols": approved,
        "hard_jump_threshold": HARD_JUMP,
        "review_jump_threshold": REVIEW_JUMP,
        "all_symbols_have_clean_source": sources_ok,
        "cross_source_returns_passed": compare_ok,
        "canonical_passed": canonical_ok,
        "synchronized": synced,
        "rule": "Any missing file, invalid OHLC, duplicate date or absolute daily close jump above 20.5% blocks the entire pool. Moves above 11.5% are separately reported and require independent-source return agreement.",
    }
    pd.concat(source_audits, ignore_index=True).to_csv(
        OUT / "source_audit.csv", index=False, encoding="utf-8-sig"
    )
    source_plan.to_csv(OUT / "source_plan.csv", index=False, encoding="utf-8-sig")
    canonical_audit.to_csv(OUT / "canonical_audit.csv", index=False, encoding="utf-8-sig")
    canonical_events.to_csv(
        OUT / "canonical_events.csv", index=False, encoding="utf-8-sig"
    )
    comparison.to_csv(OUT / "cross_source.csv", index=False, encoding="utf-8-sig")
    all_events.to_csv(OUT / "review_events.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(OUT / "manifest.csv", index=False, encoding="utf-8-sig")
    STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    # S has a larger research universe. Core symbols always use the audited C
    # canonical file. Extended symbols are admitted only when their own file
    # has no structural error or impossible price jump; failures are excluded.
    s_pool = pd.read_csv(S_POOL, dtype={"symbol": str}, encoding="utf-8-sig")
    s_enabled = s_pool[
        s_pool["enabled"].astype(str).str.lower().isin(["true", "1", "yes"])
    ].copy()
    s_symbols = s_enabled["symbol"].astype(str).str.zfill(6).tolist()
    s_audit, s_events = audit_directory(s_symbols, STAGING, "S_staging")
    s_audit_map = s_audit.set_index("symbol").to_dict("index")
    canonical_approved = set(approved)
    s_plan_rows = []
    s_approved = []
    for symbol in s_symbols:
        if symbol in canonical_approved:
            selected_source = "C_canonical"
            selected_path = CANONICAL / f"{symbol}.csv"
            symbol_passed = True
        elif bool(s_audit_map[symbol]["passed"]):
            selected_source = "S_staging"
            selected_path = STAGING / f"{symbol}.csv"
            symbol_passed = True
        else:
            selected_source = "EXCLUDED"
            selected_path = STAGING / f"{symbol}.csv"
            symbol_passed = False
        if symbol_passed:
            s_approved.append(symbol)
        s_plan_rows.append(
            {
                "symbol": symbol,
                "selected_source": selected_source,
                "selected_path": str(selected_path),
                "passed": symbol_passed,
                "reason": "approved" if symbol_passed else s_audit_map[symbol].get("reason", "failed"),
            }
        )
    s_plan = pd.DataFrame(s_plan_rows)
    s_excluded = s_plan.loc[~s_plan["passed"], "symbol"].tolist()
    staging_failed = s_audit.loc[~s_audit["passed"], "symbol"].tolist()
    if args.sync and staging_failed:
        S_QUARANTINE.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for symbol in staging_failed:
            source = STAGING / f"{symbol}.csv"
            if not source.exists():
                continue
            target = S_QUARANTINE / f"{symbol}_{stamp}.csv"
            shutil.move(str(source), str(target))
            s_plan.loc[s_plan["symbol"].eq(symbol), "quarantined_path"] = str(target)
    s_gate_passed = bool(overall and canonical_approved.issubset(set(s_approved)))
    s_status = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "passed": s_gate_passed,
        "pool_symbols": len(s_symbols),
        "approved_count": len(s_approved),
        "approved_symbols": s_approved,
        "excluded_symbols": s_excluded,
        "quarantined_staging_symbols": staging_failed,
        "core_canonical_symbols": len(canonical_approved),
        "hard_jump_threshold": HARD_JUMP,
        "review_jump_threshold": REVIEW_JUMP,
        "rule": "Core symbols use C canonical data. Extended symbols with structural errors or absolute daily close jumps above 20.5% are excluded from every strategy read.",
    }
    S_OUT.mkdir(parents=True, exist_ok=True)
    s_audit.to_csv(S_OUT / "source_audit.csv", index=False, encoding="utf-8-sig")
    s_events.to_csv(S_OUT / "review_events.csv", index=False, encoding="utf-8-sig")
    s_plan.to_csv(S_OUT / "source_plan.csv", index=False, encoding="utf-8-sig")
    S_STATUS.write_text(json.dumps(s_status, ensure_ascii=False, indent=2), encoding="utf-8")
    status["s_extended_gate"] = {
        "passed": s_gate_passed,
        "pool_symbols": len(s_symbols),
        "approved_count": len(s_approved),
        "excluded_symbols": s_excluded,
        "quarantined_staging_symbols": staging_failed,
    }
    artifact_rows = []
    artifact_paths = {
        "C_hybrid_formal": WORKSPACE / "C" / "fit" / "hybrid_formal" / "final_trades.csv",
        "S_selector_formal": WORKSPACE / "S" / "fit" / "selector" / "final_trades.csv",
    }
    allowed_all = set(s_approved)
    for artifact, path in artifact_paths.items():
        if not path.exists():
            artifact_rows.append(
                {"artifact": artifact, "path": str(path), "exists": False, "passed": False, "excluded_symbols": ""}
            )
            continue
        frame = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
        used = set(frame["symbol"].astype(str).str.zfill(6)) if "symbol" in frame else set()
        blocked = sorted(used - allowed_all)
        artifact_rows.append(
            {
                "artifact": artifact,
                "path": str(path),
                "exists": True,
                "passed": not blocked,
                "excluded_symbols": ",".join(blocked),
            }
        )
    artifact_audit = pd.DataFrame(artifact_rows)
    artifact_audit.to_csv(OUT / "model_artifact_audit.csv", index=False, encoding="utf-8-sig")
    status["historical_model_artifacts"] = {
        "passed": bool(artifact_audit["passed"].all()),
        "warning": "Pre-clean historical performance files are not valid as clean-data evidence when this check fails.",
        "failed_artifacts": artifact_audit.loc[~artifact_audit["passed"], "artifact"].tolist(),
    }
    STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if not overall or not s_gate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
