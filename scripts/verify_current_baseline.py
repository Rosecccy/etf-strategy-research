from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "same_day_1445" / "release_v2"
REQUIRED = [
    RELEASE / "README.md",
    RELEASE / "selected_summary.csv",
    RELEASE / "selected_choices.csv",
    RELEASE / "selected_trades_csr.csv",
    RELEASE / "d_annual_compact.csv",
    RELEASE / "audit.json",
]
EXPECTED = {"C": 38, "S": 44, "D": 1303, "R": 44}

missing = [str(p.relative_to(ROOT)) for p in REQUIRED if not p.exists()]
if missing:
    raise SystemExit("MISSING: " + ", ".join(missing))

with (RELEASE / "selected_summary.csv").open("r", encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))

line_key = next((k for k in ("line", "strategy", "strategy_line") if rows and k in rows[0]), None)
if line_key is None:
    raise SystemExit("selected_summary.csv has no strategy-line column")
found = {str(r[line_key]).strip() for r in rows}
if found != set(EXPECTED):
    raise SystemExit(f"strategy lines mismatch: {sorted(found)}")

audit = json.loads((RELEASE / "audit.json").read_text(encoding="utf-8-sig"))
for line, expected_rows in EXPECTED.items():
    info = audit.get("lines", {}).get(line, {})
    if info.get("trade_rows") != expected_rows:
        raise SystemExit(f"{line} trade_rows mismatch: {info.get('trade_rows')} != {expected_rows}")
    if info.get("all_year_choices_are_causal") is not True:
        raise SystemExit(f"{line} yearly choices are not causal")
if audit.get("ok") is not True:
    raise SystemExit("release audit is not OK")

print("CURRENT BASELINE OK: C/S/D/R present, 1429 audited trades, causal yearly choices")
