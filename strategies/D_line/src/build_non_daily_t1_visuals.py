from __future__ import annotations

"""Build auditable trade logs and interactive price charts for corrected D tests."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import factor_dca_scan as dca
import gate_nested_oos as gate_study
from ma120_trade_audit import build_trade_log


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
OUT = ROOT / "out"
TEMPLATE = Path(__file__).with_name("ma120_trade_template.html")
VIS_DIR = Path(r"C:\Users\10619\.codex\visualizations\2026\06\21\019ee960-4ad4-7931-9173-60478085fa8b")
TARGETS = {
    "winrate": {"gate": "self_ma120", "title": "D线：高胜率策略（自身站上120日均线）"},
    "return": {"gate": "market_ret20", "title": "D线：高累计收益策略（市场近20日收益非负）"},
}


def candidate_from_label(
    combo: str,
    conditions: dict[str, np.ndarray],
    labels: dict[str, str],
) -> dca.Candidate:
    reverse = {label: identifier for identifier, label in labels.items()}
    identifiers = tuple(reverse[item] for item in combo.split(" + "))
    mask = np.logical_and.reduce([conditions[item] for item in identifiers])
    return dca.Candidate(" + ".join(identifiers), combo, identifiers, mask)


def chart_payload(trades: pd.DataFrame, summary: dict) -> dict:
    trades = trades.copy()
    trades["symbol"] = trades["symbol"].astype(str).str.zfill(6)
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"], errors="coerce")
    year_symbols: dict[int, set[str]] = {}
    for _, row in trades.iterrows():
        year_symbols.setdefault(int(row["entry_date"].year), set()).add(row["symbol"])
        if pd.notna(row["exit_date"]):
            year_symbols.setdefault(int(row["exit_date"].year), set()).add(row["symbol"])

    by_year: dict[str, dict] = {}
    raw_dir = PROJECT / "C" / "raw" / "etf"
    for year, symbols in sorted(year_symbols.items()):
        by_year[str(year)] = {}
        for symbol in sorted(symbols):
            related = trades[
                (trades["symbol"].eq(symbol))
                & ((trades["entry_date"].dt.year.eq(year)) | (trades["exit_date"].dt.year.eq(year)))
            ].copy()
            raw = pd.read_csv(raw_dir / f"{symbol}.csv", encoding="utf-8-sig")
            raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
            raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
            raw = raw[(raw["date"].dt.year.eq(year)) & raw["close"].notna()].sort_values("date")
            if raw.empty:
                continue
            buys = related[related["entry_date"].dt.year.eq(year)]
            sells = related[related["exit_date"].dt.year.eq(year)]
            records = []
            for _, row in related.iterrows():
                records.append(
                    {
                        "entry_date": row["entry_date"].date().isoformat(),
                        "exit_date": row["exit_date"].date().isoformat() if pd.notna(row["exit_date"]) else "",
                        "entry_close": round(float(row["entry_close"]), 6),
                        "exit_close": round(float(row["exit_close"]), 6) if pd.notna(row["exit_close"]) else None,
                        "return_rate": round(float(row["return_rate"]), 8) if pd.notna(row["return_rate"]) else None,
                        "status": str(row["status"]),
                    }
                )
            by_year[str(year)][symbol] = {
                "symbol": symbol,
                "name": str(related.iloc[0]["name"]),
                "prices": [
                    {"date": row.date.date().isoformat(), "close": round(float(row.close), 6)}
                    for row in raw[["date", "close"]].itertuples(index=False)
                ],
                "buys": [
                    {"date": row.entry_date.date().isoformat(), "price": round(float(row.entry_close), 6)}
                    for row in buys.itertuples(index=False)
                ],
                "sells": [
                    {"date": row.exit_date.date().isoformat(), "price": round(float(row.exit_close), 6)}
                    for row in sells.itertuples(index=False)
                ],
                "trades": records,
            }
    return {"summary": summary, "years": list(by_year), "byYear": by_year}


def main() -> None:
    selected = pd.read_csv(OUT / "non_daily_t1_gate_years.csv", encoding="utf-8-sig")
    full, groups = dca.load_panel(non_daily_signal_on_period_close=True)
    buy_conditions, buy_labels, _ = dca.build_conditions(full, "buy")
    sell_conditions, sell_labels, _ = dca.build_conditions(full, "sell")
    all_gates = gate_study.gates(full)
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count("__DATA__") != 1:
        raise RuntimeError("Visualization template must contain exactly one data marker.")
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    for tag, target in TARGETS.items():
        rows = selected[(selected["gate"].eq(target["gate"])) & (selected["status"].eq("selected"))].copy()
        if rows.empty:
            raise RuntimeError(f"No selected rolling rows for {target['gate']}.")
        gated_buy_conditions = {key: value & all_gates[target["gate"]] for key, value in buy_conditions.items()}
        logs: list[pd.DataFrame] = []
        for _, row in rows.iterrows():
            year = int(row["test_year"])
            buy = candidate_from_label(str(row["buy_combo"]), gated_buy_conditions, buy_labels)
            sell = candidate_from_label(str(row["sell_combo"]), sell_conditions, sell_labels)
            logs.append(
                build_trade_log(
                    buy,
                    sell,
                    full,
                    groups,
                    pd.Timestamp(year=year, month=1, day=1),
                    pd.Timestamp(year=year + 1, month=1, day=1),
                    year,
                    str(row["rule"]),
                )
            )
        trades = pd.concat(logs, ignore_index=True)
        closed = trades[trades["status"].eq("closed")].copy()
        expected_closed = int(rows["test_closed_batches"].sum())
        expected_wins = int(rows["test_win_batches"].sum())
        actual_wins = int((closed["return_rate"] > 0).sum())
        if len(closed) != expected_closed or actual_wins != expected_wins:
            raise RuntimeError(
                f"Audit mismatch for {tag}: expected {expected_closed}/{expected_wins}, "
                f"got {len(closed)}/{actual_wins}."
            )
        summary = {
            "title": target["title"],
            "gate": target["gate"],
            "closed_batches": int(len(closed)),
            "win_batches": actual_wins,
            "win_rate": float((closed["return_rate"] > 0).mean()),
            "average_return": float(closed["return_rate"].mean()),
            "realized_pnl_cny": float(closed["pnl_cny"].sum()),
            "execution": "Non-daily period-end close confirms the signal; execution is at the next trading-day close. Daily signals remain T+1 close.",
        }
        trades.to_csv(OUT / f"non_daily_t1_{tag}_trade_log.csv", index=False, encoding="utf-8-sig")
        (OUT / f"non_daily_t1_{tag}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        payload = json.dumps(chart_payload(trades, summary), ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
        fragment = template.replace("__DATA__", payload)
        fragment_path = VIS_DIR / f"non-daily-t1-{tag}.html"
        fragment_path.write_text(fragment, encoding="utf-8")
        (OUT / f"non_daily_t1_{tag}_fragment.html").write_text(fragment, encoding="utf-8")
        print(json.dumps({"tag": tag, **summary, "fragment": str(fragment_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
