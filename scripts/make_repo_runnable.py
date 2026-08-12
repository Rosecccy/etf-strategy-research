from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ROOT / "strategies"


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def clear_runtime_dir(name: str) -> None:
    target = ROOT / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def build_line(line: str, src_name: str) -> None:
    src = STRATEGIES / src_name
    dst = ROOT / line
    clear_runtime_dir(line)

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
        # V shares several panic/fear-greed helpers with D-line code.
        copy_tree(STRATEGIES / "D_line" / "src", dst / "src")
        copy_tree(src / "results" / "fear_greed", dst / "out" / "fear_greed")
        copy_tree(src / "results" / "forced_top1_confidence_gate", dst / "out" / "forced_top1_confidence_gate")
        copy_tree(src / "results" / "frequency_preserving_upgrade", dst / "out" / "frequency_preserving_upgrade")


def first_csv_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return next(reader, {})


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_runner() -> None:
    code = r'''from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def first_csv_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return next(reader, {})


def line_summary() -> dict:
    return {
        "C_line": {
            "today_decision": first_csv_row(ROOT / "C" / "live" / "today_decision.csv"),
            "summary": read_json(ROOT / "C" / "live" / "summary.json"),
        },
        "S_line": {
            "today_decision": first_csv_row(ROOT / "S" / "live" / "today_decision.csv"),
            "top_shadow_candidate": first_csv_row(ROOT / "S" / "live" / "today_shadow_candidates.csv"),
            "model_state": read_json(ROOT / "S" / "live" / "model_state.json"),
        },
        "D_line": {
            "today": first_csv_row(ROOT / "D" / "live" / "today.csv"),
            "today_json": read_json(ROOT / "D" / "live" / "today.json"),
            "panic_today": read_json(ROOT / "D" / "live" / "daily_panic_today.json"),
        },
        "V_line": {
            "fear_greed_summary": read_json(ROOT / "V" / "out" / "fear_greed" / "summary.json"),
            "forced_top1_summary": first_csv_row(ROOT / "V" / "out" / "forced_top1_confidence_gate" / "summary.csv"),
        },
    }


if __name__ == "__main__":
    print(json.dumps(line_summary(), ensure_ascii=False, indent=2))
'''
    (ROOT / "run_all_strategies.py").write_text(code, encoding="utf-8")


def write_verify() -> None:
    code = r'''from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REQUIRED = [
    ROOT / "C" / "raw" / "etf",
    ROOT / "C" / "live" / "today_decision.csv",
    ROOT / "S" / "raw" / "etf",
    ROOT / "S" / "live" / "today_decision.csv",
    ROOT / "D" / "raw" / "etf",
    ROOT / "D" / "live" / "today.csv",
    ROOT / "V" / "out" / "fear_greed",
    ROOT / "run_all_strategies.py",
]


def main() -> int:
    bootstrap = ROOT / "scripts" / "make_repo_runnable.py"
    if bootstrap.exists():
        subprocess.run([sys.executable, str(bootstrap)], cwd=ROOT, check=True)

    missing = [str(p.relative_to(ROOT)) for p in REQUIRED if not p.exists()]
    if missing:
        print("PACKAGE NOT RUNNABLE")
        print(json.dumps({"missing": missing}, ensure_ascii=False, indent=2))
        return 1

    result = subprocess.run(
        [sys.executable, str(ROOT / "run_all_strategies.py")],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )
    if result.returncode != 0:
        print("RUNNER FAILED")
        print(result.stdout)
        print(result.stderr)
        return result.returncode

    print("PACKAGE RUNNABLE")
    print(result.stdout)

    deep_checks = [
        ["C", [sys.executable, str(ROOT / "C" / "src" / "run_daily.py"), "--skip-update"]],
        ["S", [sys.executable, str(ROOT / "S" / "src" / "run_s1_live.py")]],
        ["D", [sys.executable, str(ROOT / "D" / "src" / "daily_panic_live.py")]],
    ]
    for name, cmd in deep_checks:
        check = subprocess.run(cmd, cwd=ROOT, text=True, encoding="utf-8", capture_output=True)
        if check.returncode != 0:
            print(f"{name}_LINE_ENTRY_FAILED")
            print(check.stdout)
            print(check.stderr)
            return check.returncode
        print(f"{name}_LINE_ENTRY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    (ROOT / "scripts" / "verify_runnable.py").write_text(code, encoding="utf-8")


def update_readme() -> None:
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    addon = """

## 离线运行检查

这个仓库现在同时保留两套结构：

- `strategies/C_line`、`strategies/S_line`、`strategies/D_line`、`strategies/V_line`：给人阅读和复核的精简结构。
- `C`、`S`、`D`、`V`：本地运行时自动生成的兼容结构，路径名与原工程一致；这些目录不需要提交到 Git。

同学拿到仓库后，先在仓库根目录运行：

```bash
python scripts/verify_package.py
python scripts/verify_runnable.py
python run_all_strategies.py
```

其中 `verify_runnable.py` 会先自动生成本地兼容目录，再在不依赖旧电脑目录的情况下读取本仓库内的数据、配置和结果，并实际调用 C/S/D 三条策略入口。
"""
    if "## 离线运行检查" not in text:
        readme.write_text(text.rstrip() + "\n" + addon, encoding="utf-8")


def main() -> None:
    build_line("C", "C_line")
    build_line("S", "S_line")
    build_line("D", "D_line")
    build_line("V", "V_line")
    write_runner()
    write_verify()
    update_readme()
    print("runnable compatibility layer built")


if __name__ == "__main__":
    main()
