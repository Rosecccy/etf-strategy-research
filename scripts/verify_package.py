from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "requirements.txt",
    "docs/REPRODUCE.md",
    "docs/GITHUB_PRIVATE_SETUP.md",
    "strategies/C_line/README.md",
    "strategies/C_line/results/clean_upgrades/summary.json",
    "strategies/S_line/README.md",
    "strategies/S_line/results/selector/summary.json",
    "strategies/D_line/README.md",
    "strategies/D_line/results/formal/current_strategy.json",
    "strategies/D_line/results/formal/formal_selected_trades.csv",
    "strategies/V_line/README.md",
    "strategies/V_line/results/fear_greed/summary.json",
    "file_manifest_sha256.csv",
    "PACKAGE_AUDIT.json",
]

def main() -> int:
    ok = True
    for rel in REQUIRED:
        p = ROOT / rel
        if not p.exists():
            print(f"MISSING: {rel}")
            ok = False

    for line in ["C_line", "S_line", "D_line", "V_line"]:
        data_dir = ROOT / "strategies" / line / "data" / "etf"
        csv_count = len(list(data_dir.glob("*.csv"))) if data_dir.exists() else 0
        print(f"{line}: ETF csv files = {csv_count}")
        if csv_count == 0:
            ok = False

    too_big = [p for p in ROOT.rglob("*") if p.is_file() and p.stat().st_size > 100 * 1024 * 1024]
    for p in too_big:
        print(f"OVER_100MB: {p.relative_to(ROOT)}")
        ok = False

    if ok:
        print("PACKAGE OK")
        return 0
    print("PACKAGE FAILED")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
