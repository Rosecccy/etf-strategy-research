"""Build the historical C/S/D/V runtime compatibility folders.

This helper exists only for the legacy `strategies/` layer. It does not build,
change, or validate the current C/S/D/R 2026-08-21 V2 baseline and it never
edits the repository README.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ROOT / "strategies"


def copy_tree(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copytree(
            src,
            dst,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )


def copy_file(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def clear_runtime_dir(name: str) -> Path:
    target = ROOT / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def build_line(line: str, src_name: str) -> None:
    src = STRATEGIES / src_name
    dst = clear_runtime_dir(line)
    copy_tree(src / "src", dst / "src")
    copy_tree(src / "data" / "etf", dst / "raw" / "etf")
    copy_tree(src / "data" / "quality", dst / "raw" / "quality")
    copy_file(src / "data" / "pool.csv", dst / "raw" / "pool.csv")
    copy_file(src / "data" / "quality.json", dst / "raw" / "quality.json")
    copy_tree(src / "config", dst / "cfg")
    copy_tree(src / "live", dst / "live")

    if line == "C":
        copy_tree(src / "results" / "clean_upgrades", dst / "fit" / "clean_upgrades")
        copy_tree(src / "results" / "extrema", dst / "fit" / "extrema")
        copy_tree(src / "results" / "selector", dst / "fit" / "selector")
        copy_tree(src / "results" / "fallback", dst / "fit" / "fallback")
    elif line == "S":
        copy_tree(src / "data" / "pool_detail", dst / "pool")
        copy_tree(src / "results" / "base", dst / "fit" / "base")
        copy_tree(src / "results" / "selector", dst / "fit" / "selector")
        copy_tree(src / "results" / "strengthen", dst / "fit" / "strengthen")
    elif line == "D":
        copy_tree(src / "results" / "formal", dst / "formal")
        copy_tree(src / "results" / "csd_v2_exit_guard", dst / "out" / "csd_v2_exit_guard")
        copy_tree(src / "results" / "csd_regime_pocket", dst / "out" / "csd_regime_pocket")
    elif line == "V":
        copy_tree(STRATEGIES / "D_line" / "src", dst / "src")
        copy_tree(src / "results" / "fear_greed", dst / "out" / "fear_greed")
        copy_tree(src / "results" / "forced_top1_confidence_gate", dst / "out" / "forced_top1_confidence_gate")
        copy_tree(src / "results" / "frequency_preserving_upgrade", dst / "out" / "frequency_preserving_upgrade")


def main() -> None:
    build_line("C", "C_line")
    build_line("S", "S_line")
    build_line("D", "D_line")
    build_line("V", "V_line")
    print("LEGACY C/S/D/V compatibility folders built; current C/S/D/R V2 unchanged")


if __name__ == "__main__":
    main()
