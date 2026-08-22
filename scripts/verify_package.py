from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "file_manifest_sha256.csv"
PACKAGE_AUDIT = ROOT / "PACKAGE_AUDIT.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest() -> list[str]:
    errors: list[str] = []
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return ["manifest is empty"]
    for row in rows:
        rel = row["path"]
        path = ROOT / rel
        if not path.exists():
            errors.append(f"manifest missing file: {rel}")
            continue
        expected_size = int(row["size_bytes"])
        if path.stat().st_size != expected_size:
            errors.append(f"manifest size mismatch: {rel}")
            continue
        if sha256(path) != row["sha256"]:
            errors.append(f"manifest hash mismatch: {rel}")
    return errors


def canonical_required_paths() -> list[Path]:
    return [
        ROOT / "README.md",
        ROOT / "baseline.json",
        ROOT / "docs" / "REPRODUCE.md",
        ROOT / "docs" / "DATA_RECONSTRUCTION.md",
        ROOT / "docs" / "OPTIMIZATION_PROTOCOL.md",
        ROOT / "docs" / "REPOSITORY_SCOPE.md",
        ROOT / "scripts" / "baseline_audit.py",
        ROOT / "scripts" / "verify_current_baseline.py",
        ROOT / "same_day_1445" / "release_v2" / "selected_trades_csr.csv",
        ROOT / "same_day_1445" / "release_v2" / "d_annual_compact.csv",
        ROOT / "same_day_1445" / "release_v2" / "d_cross_year_entry_audit.csv",
        MANIFEST,
        PACKAGE_AUDIT,
    ]


def main() -> int:
    required = canonical_required_paths()
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        print("PACKAGE FAILED")
        print("missing:", ", ".join(missing))
        return 1

    current = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_current_baseline.py")],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )
    if current.returncode != 0:
        print("PACKAGE FAILED: current V2 baseline verification")
        print(current.stdout)
        print(current.stderr)
        return current.returncode

    audit = json.loads(PACKAGE_AUDIT.read_text(encoding="utf-8-sig"))
    if audit.get("canonical_baseline") != "same_day_1445/release_v2":
        print("PACKAGE FAILED: PACKAGE_AUDIT.json canonical baseline mismatch")
        return 1
    if audit.get("canonical_lines") != ["C", "S", "D", "R"]:
        print("PACKAGE FAILED: PACKAGE_AUDIT.json line definition mismatch")
        return 1

    manifest_errors = verify_manifest()
    if manifest_errors:
        print("PACKAGE FAILED: manifest")
        for error in manifest_errors:
            print("-", error)
        return 1

    legacy = [ROOT / "strategies" / name for name in ("C_line", "S_line", "D_line", "V_line")]
    legacy_present = all(path.exists() for path in legacy)

    print(current.stdout.rstrip())
    print("PACKAGE OK")
    print("- canonical V2 audit surface hashes verified")
    print(f"- legacy C/S/D/V compatibility layer present: {legacy_present}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
