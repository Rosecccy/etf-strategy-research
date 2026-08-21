from __future__ import annotations

import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "ledger_audit"
SOURCES = {
    "C_formal": ROOT / "C/formal/c_trades.csv",
    "C_formal_fallback": ROOT / "C/formal/fallback/final_trades.csv",
    "C_legacy_selected": ROOT / "C/formal/live_rules/best_selected_trades.csv",
    "C_legacy_best": ROOT / "C/formal/live_rules/best_trades.csv",
    "C_suspicious_s_trades": ROOT / "C/formal/s_trades.csv",
    "S_formal": ROOT / "S/formal/winner_trades.csv",
    "S_seed_baseline": ROOT / "S/formal/base/domestic_baseline_account.csv",
    "S_seed_trades": ROOT / "S/formal/base/s1_seed_trades.csv",
    "D_formal": ROOT / "D/formal/formal_selected_trades.csv",
    "D_panic_research": ROOT / "D/formal/panic/base_trades.csv",
}

def read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    frame.columns = [str(c).strip() for c in frame.columns]
    if "symbol" in frame:
        frame["symbol"] = frame["symbol"].astype(str).str.extract(r"(\d{6})", expand=False)
    elif "code" in frame:
        frame["symbol"] = frame["code"].astype(str).str.extract(r"(\d{6})", expand=False)
    else:
        frame["symbol"] = ""
    for column in ("entry_date", "signal_date", "decision_date", "exit_date", "exit_date_new", "original_exit_date"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame

def date_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in frame:
            return name
    return None

def normalized(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    result = frame.copy()
    entry = date_column(result, ("entry_date",))
    exit_ = date_column(result, ("exit_date", "exit_date_new", "original_exit_date"))
    ret = date_column(result, ("ret", "return", "ret_new", "return_rate"))
    result["entry_key"] = result[entry].dt.strftime("%Y-%m-%d") if entry else ""
    result["exit_key"] = result[exit_].dt.strftime("%Y-%m-%d") if exit_ else ""
    result["ret_value"] = pd.to_numeric(result[ret], errors="coerce") if ret else pd.NA
    result["ledger_key"] = result["symbol"].fillna("") + "|" + result["entry_key"].fillna("")
    return result

def classify(label: str) -> str:
    if label.endswith("_formal") or label == "D_formal": return "正式候选账本"
    if "legacy" in label: return "旧版本研究表"
    if "seed" in label or "panic" in label or "suspicious" in label: return "输入/研究对照表"
    return "其他"

def pair_report(left_name: str, left: pd.DataFrame, right_name: str, right: pd.DataFrame) -> dict:
    l = left.drop_duplicates("ledger_key").set_index("ledger_key")[["exit_key", "ret_value"]]
    r = right.drop_duplicates("ledger_key").set_index("ledger_key")[["exit_key", "ret_value"]]
    compare = l.join(r, how="inner", lsuffix="_left", rsuffix="_right")
    same_exit = int(compare["exit_key_left"].eq(compare["exit_key_right"]).sum())
    left_ret = pd.to_numeric(compare["ret_value_left"], errors="coerce")
    right_ret = pd.to_numeric(compare["ret_value_right"], errors="coerce")
    same_ret = int((left_ret.notna() & right_ret.notna() & (left_ret - right_ret).abs().le(1e-10)).sum())
    return {"left": left_name,"right": right_name,"left_rows": len(left),"right_rows": len(right),"left_duplicate_keys": int(left["ledger_key"].duplicated(keep=False).sum()),"right_duplicate_keys": int(right["ledger_key"].duplicated(keep=False).sum()),"overlap_keys": len(compare.index),"same_exit_date": same_exit,"exit_conflicts": int(len(compare.index) - same_exit),"same_return": same_ret,"return_conflicts": int(len(compare.index) - same_ret)}

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    manifest = []
    duplicate_rows = []
    for label, path in SOURCES.items():
        exists = path.exists()
        manifest.append({"label": label, "path": str(path.relative_to(ROOT)), "exists": exists, "role": classify(label)})
        if not exists: continue
        frame = normalized(read(path), label); frames[label] = frame
        duplicate = frame[frame["ledger_key"].duplicated(keep=False)].copy()
        if not duplicate.empty:
            duplicate.insert(0, "source", label)
            duplicate_rows.append(duplicate[["source", "ledger_key", "symbol", "entry_key", "exit_key", "ret_value"]])
        manifest[-1].update({"rows": len(frame),"unique_entry_keys": int(frame["ledger_key"].nunique()),"duplicate_key_rows": int(frame["ledger_key"].duplicated(keep=False).sum())})
    pair_specs = [("C_formal", "C_formal_fallback"),("C_formal", "C_legacy_selected"),("S_formal", "S_seed_baseline"),("S_formal", "S_seed_trades"),("D_formal", "D_panic_research")]
    pairs = [pair_report(a, frames[a], b, frames[b]) for a, b in pair_specs if a in frames and b in frames]
    pd.DataFrame(manifest).to_csv(OUT / "ledger_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pairs).to_csv(OUT / "ledger_pairwise.csv", index=False, encoding="utf-8-sig")
    pd.concat(duplicate_rows, ignore_index=True).to_csv(OUT / "duplicate_keys.csv", index=False, encoding="utf-8-sig") if duplicate_rows else pd.DataFrame().to_csv(OUT / "duplicate_keys.csv", index=False, encoding="utf-8-sig")
    c_conflicts = next((p for p in pairs if p["left"] == "C_formal" and p["right"] == "C_formal_fallback"), {})
    formal_rows = {label: next((row for row in manifest if row["label"] == label), {}) for label in ("C_formal", "S_formal", "D_formal")}
    formal_internal_ok = all(bool(row.get("exists")) and int(row.get("duplicate_key_rows", 1)) == 0 for row in formal_rows.values())
    summary = {"formal_ledger": {"C": "C/formal/c_trades.csv","S": "S/formal/winner_trades.csv","D": "D/formal/formal_selected_trades.csv"},"C_formal_vs_fallback_exit_conflicts": c_conflicts.get("exit_conflicts", None),"C_formal_vs_fallback_return_conflicts": c_conflicts.get("return_conflicts", None),"duplicate_key_source_count": sum(1 for row in manifest if row.get("duplicate_key_rows", 0) > 0),"formal_tables_have_internal_duplicate_keys": {line: bool(next((row.get("duplicate_key_rows", 0) for row in manifest if row["label"] == label), 0)) for line, label in (("C", "C_formal"), ("S", "S_formal"), ("D", "D_formal"))},"formal_ledgers_internal_consistent": formal_internal_ok,"research_table_differences_are_formal_conflicts": False,"formal_promotion_allowed": formal_internal_ok,"note": "C备用表、S种子表、D恐惧因子表只作研究对照，不参与正式账本判断。"}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
