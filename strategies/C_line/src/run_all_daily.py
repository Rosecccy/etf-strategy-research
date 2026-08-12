from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "C" / "live" / "system"


def configure_stdout_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def run_cmd(args: list[str], timeout: int = 900) -> dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=timeout,
    )
    return {
        "cmd": " ".join(args),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-3000:],
        "stderr_tail": proc.stderr[-3000:],
    }


def latest_report(line: str, target_date: str) -> pd.DataFrame:
    etf_dir = ROOT / line / "raw" / "etf"
    rows = []
    for path in sorted(etf_dir.glob("*.csv")):
        try:
            df = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
            latest = pd.to_datetime(df["date"], errors="coerce").max()
            rows.append(
                {
                    "line": line,
                    "symbol": path.stem,
                    "latest_date": latest.strftime("%Y-%m-%d") if pd.notna(latest) else "",
                    "rows": int(len(df)),
                    "is_stale": bool(pd.notna(latest) and latest.strftime("%Y-%m-%d") < target_date),
                }
            )
        except Exception as exc:
            rows.append({"line": line, "symbol": path.stem, "latest_date": "", "rows": 0, "is_stale": True, "error": repr(exc)})
    return pd.DataFrame(rows)


def stale_symbols(target_date: str) -> list[str]:
    report = latest_report("S", target_date)
    if report.empty:
        return []
    return report.loc[report["is_stale"], "symbol"].astype(str).str.zfill(6).tolist()


def write_reports(target_date: str, steps: list[dict]) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    reports = []
    for line in ("C", "S"):
        report = latest_report(line, target_date)
        report.to_csv(OUT / f"{line.lower()}_freshness.csv", index=False, encoding="utf-8-sig")
        reports.append(report)
    all_reports = pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()
    all_reports.to_csv(OUT / "freshness.csv", index=False, encoding="utf-8-sig")
    quality_path = ROOT / "C" / "raw" / "quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {"passed": False}
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_date": target_date,
        "steps": steps,
        "data_quality": quality,
        "freshness": {
            line: {
                "total": int((all_reports["line"] == line).sum()) if not all_reports.empty else 0,
                "stale": int(((all_reports["line"] == line) & (all_reports["is_stale"])).sum()) if not all_reports.empty else 0,
            }
            for line in ("C", "S")
        },
    }
    (OUT / "daily_status.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    configure_stdout_utf8()
    parser = argparse.ArgumentParser(description="Unified C/S data update, validation and decision generation.")
    parser.add_argument("--target-date", default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--retry-stale", action="store_true")
    args = parser.parse_args()
    target_compact = args.target_date.replace("-", "")
    py = sys.executable
    steps = [run_cmd([py, "C/src/init_state.py", "--asof", args.target_date], timeout=60)]

    if not args.skip_download:
        steps.append(run_cmd([py, "C/src/update_data.py", "--end", target_compact], timeout=1200))
        steps.append(run_cmd([py, "S/src/update_data.py", "--end", target_compact, "--delay", "0", "--retries", "0"], timeout=1200))

    if args.retry_stale and not args.skip_download:
        for source in ("eastmoney", "tencent", "akshare"):
            stale = stale_symbols(args.target_date)
            if not stale:
                break
            steps.append(run_cmd([py, "S/src/update_data.py", "--symbols", *stale, "--end", target_compact, "--source", source, "--delay", "0", "--retries", "0"], timeout=900))

    steps.append(run_cmd([py, "S/src/repair_splits.py"], timeout=300))
    quality_step = run_cmd([py, "C/src/data_quality_gate.py", "--sync"], timeout=300)
    steps.append(quality_step)
    if quality_step["returncode"] != 0:
        print(json.dumps(write_reports(args.target_date, steps), ensure_ascii=False, indent=2))
        raise SystemExit("ETF data quality gate failed; all C/S/D decisions were blocked.")
    steps.append(run_cmd([py, "C/src/run_daily.py", "--skip-update"], timeout=1200))
    steps.append(run_cmd([py, "S/src/run_s1_live.py"], timeout=600))
    steps.append(run_cmd([py, "C/src/hybrid_live.py"], timeout=120))
    steps.append(run_cmd([py, "D/src/daily_panic_live.py"], timeout=600))
    steps.append(run_cmd([py, "D/src/build_unified_site.py"], timeout=120))
    print(json.dumps(write_reports(args.target_date, steps), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
