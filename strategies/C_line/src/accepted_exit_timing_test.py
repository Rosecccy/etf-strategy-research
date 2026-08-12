from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import controlled_upgrade_search as stats_mod
import take_profit_10_test as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit" / "accepted_exit_timing"

SOURCES = {
    "S": ROOT / "fit" / "trend_extension" / "s_trades.csv",
    "D": ROOT / "fit" / "novel_path_exits" / "d_trades.csv",
}
PROJECTS = {"S": ROOT.parent / "S", "D": ROOT}


def load(line: str) -> pd.DataFrame:
    frame = pd.read_csv(SOURCES[line], encoding="utf-8-sig", dtype={"symbol": str})
    frame["symbol"] = frame["symbol"].str.zfill(6)
    for column in ("entry_date", "exit_date_test", "extend_signal_date", "path_signal_date"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def apply_mode(
    line: str,
    frame: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    mode: str,
) -> pd.DataFrame:
    trigger_col = "extend_triggered" if line == "S" else "path_triggered"
    signal_col = "extend_signal_date" if line == "S" else "path_signal_date"
    rows = []
    for _, trade in frame.iterrows():
        item = trade.to_dict()
        triggered = bool(trade.get(trigger_col, False)) and pd.notna(trade.get(signal_col))
        item["exit_timing_mode"] = mode
        item["same_day_exit_used"] = False
        if triggered:
            signal = pd.Timestamp(trade[signal_col])
            data = prices[str(trade["symbol"])]
            row = data[data["date"].eq(signal)]
            if not row.empty:
                use_same_day = mode == "same_day"
                if mode.startswith("vol_state"):
                    window = int(mode.split("_")[-1])
                    pos = int(pd.DatetimeIndex(data["date"]).searchsorted(signal, side="left"))
                    history = data.iloc[max(0, pos - 20):pos + 1]["close"].pct_change().dropna()
                    annual_vol = float(history.std(ddof=0) * np.sqrt(244)) if len(history) >= 10 else 0.0
                    threshold = window / 100.0
                    use_same_day = annual_vol >= threshold
                if use_same_day:
                    item["exit_date_test"] = signal
                    item["exit_close_test"] = float(row.iloc[0]["close"])
                    item["exit_reason_test"] = f"{trade['exit_reason_test']}|same_day"
                    item["gross_return_test"] = float(item["exit_close_test"]) / float(item["entry_close"]) - 1.0
                    item["same_day_exit_used"] = True
        rows.append(item)
    return pd.DataFrame(rows)


def evaluate(line: str) -> dict:
    frame = load(line)
    prices = core.load_prices(PROJECTS[line], set(frame["symbol"]))
    modes = ["next_day", "same_day"] + [f"vol_state_{value}" for value in (15, 20, 25, 30, 35, 40)]
    variants = {mode: apply_mode(line, frame, prices, mode) for mode in modes}
    base = {period: stats_mod.subset_stats(frame, period) for period in ("full", "holdout")}
    rows = []
    for mode, candidate in variants.items():
        current = {period: stats_mod.subset_stats(candidate, period) for period in ("full", "holdout")}
        rows.append(
            {
                "line": line,
                "mode": mode,
                "changed": int(candidate["same_day_exit_used"].sum()),
                "triggers": len(candidate),
                "final": current["full"]["final_value"],
                "ratio": current["full"]["final_value"] / base["full"]["final_value"],
                "win": current["full"]["win_rate"],
                "avg_annual": current["full"]["avg_annual_return"],
                "max_drawdown": current["full"]["max_drawdown"],
                "holdout_final": current["holdout"]["final_value"],
                "holdout_ratio": current["holdout"]["final_value"] / base["holdout"]["final_value"],
                "holdout_win": current["holdout"]["win_rate"],
            }
        )
        candidate.to_csv(OUT / f"{line.lower()}_{mode}_trades.csv", index=False, encoding="utf-8-sig")
    table = pd.DataFrame(rows).sort_values(["ratio", "win"], ascending=False)
    table.to_csv(OUT / f"{line.lower()}_variants.csv", index=False, encoding="utf-8-sig")
    passing = table[
        table["ratio"].gt(1.0)
        & table["holdout_ratio"].gt(1.0)
        & table["win"].ge(base["full"]["win_rate"])
        & table["holdout_win"].ge(base["holdout"]["win_rate"])
    ]
    return {
        "line": line,
        "baseline": base,
        "strict_candidates": int(len(passing)),
        "best": None if passing.empty else passing.iloc[0].to_dict(),
        "top": table.head(5).to_dict(orient="records"),
    }


def main() -> None:
    core.configure_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "execution timing for already accepted S/D exit signals",
        "constraint": "signals and trade count unchanged; same-day execution assumes decision before 15:15",
        "results": [evaluate("S"), evaluate("D")],
    }
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
