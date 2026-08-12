from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
C_RAW = ROOT / "C" / "raw" / "etf"
S_RAW = ROOT / "S" / "raw" / "etf"
V2_DIR = ROOT / "D" / "out" / "csd_regime_judge_v2_strict"
OUT_DIR = ROOT / "D" / "out" / "csd_v2_exit_guard"
BEST_SELECTOR = "all_priority_bias_min3"


def scope_applies(line: str, scope: str) -> bool:
    return scope == "all" or line in set(scope.split("+"))


def load_v2_trades() -> pd.DataFrame:
    cols = [
        "selector",
        "signal_date",
        "line",
        "symbol",
        "name",
        "entry_date",
        "exit_date",
        "ret",
        "year",
        "policy",
        "trend_score",
        "rebound_score",
    ]
    p = V2_DIR / "extra_selector_selected_trades.csv"
    df = pd.read_csv(p, usecols=cols, dtype={"symbol": str})
    df = df[df["selector"].eq(BEST_SELECTOR)].copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["year"] = df["year"].astype(int)
    df["trade_id"] = np.arange(len(df))
    return df.reset_index(drop=True)


def load_price(symbol: str) -> tuple[pd.DataFrame, str]:
    for root in [C_RAW, S_RAW]:
        p = root / f"{symbol}.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"], dtype={"symbol": str})
            df = df.sort_values("date").drop_duplicates("date")
            df = df[["date", "close"]].dropna().reset_index(drop=True)
            return df, str(p)
    raise FileNotFoundError(f"missing ETF price file: {symbol}")


def build_paths(trades: pd.DataFrame) -> tuple[dict[int, dict], pd.DataFrame]:
    prices: dict[str, pd.DataFrame] = {}
    src_rows = []
    for sym in sorted(trades["symbol"].astype(str).unique()):
        prices[sym], src = load_price(sym)
        src_rows.append(
            {
                "symbol": sym,
                "source_file": src,
                "rows": len(prices[sym]),
                "start": prices[sym]["date"].min(),
                "end": prices[sym]["date"].max(),
            }
        )

    paths: dict[int, dict] = {}
    for row in trades.itertuples(index=False):
        px = prices[str(row.symbol)]
        start = px["date"].searchsorted(row.entry_date)
        end = px["date"].searchsorted(row.exit_date, side="right") - 1
        if start >= len(px) or end < start:
            dates = np.array([], dtype="datetime64[ns]")
            closes = np.array([], dtype=float)
        else:
            part = px.iloc[start : end + 1]
            dates = part["date"].to_numpy(dtype="datetime64[ns]")
            closes = part["close"].to_numpy(dtype=float)
        paths[int(row.trade_id)] = {"dates": dates, "closes": closes}
    return paths, pd.DataFrame(src_rows)


def make_rules() -> list[dict]:
    rules: list[dict] = [{"rule_id": "base_original", "scope": "all", "exec_mode": "original"}]
    modes = ["same_close", "next_close"]
    scopes = ["all", "D", "S", "D+S"]
    for mode in modes:
        for scope in scopes:
            for tp in [0.10, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18, 0.20, 0.25]:
                rules.append({"rule_id": f"tp{tp:.0%}_{scope}_{mode}", "scope": scope, "exec_mode": mode, "tp": tp})
            for sl in [-0.04, -0.06, -0.08, -0.10]:
                rules.append({"rule_id": f"sl{abs(sl):.0%}_{scope}_{mode}", "scope": scope, "exec_mode": mode, "sl": sl})
            for cap in [20, 30, 40, 60]:
                rules.append({"rule_id": f"cap{cap}_{scope}_{mode}", "scope": scope, "exec_mode": mode, "cap": cap})

        for scope in scopes:
            for tp in [0.15, 0.20, 0.25]:
                for sl in [-0.06, -0.08, -0.10]:
                    rules.append({"rule_id": f"tp{tp:.0%}_sl{abs(sl):.0%}_{scope}_{mode}", "scope": scope, "exec_mode": mode, "tp": tp, "sl": sl})

        for scope in scopes:
            for act in [0.08, 0.12, 0.15, 0.20]:
                for dd in [0.04, 0.06, 0.08, 0.10]:
                    rules.append({"rule_id": f"trail{act:.0%}_{dd:.0%}_{scope}_{mode}", "scope": scope, "exec_mode": mode, "trail_act": act, "trail_dd": dd})

        for scope in ["all", "D", "D+S"]:
            for act in [0.08, 0.12, 0.15]:
                for dd in [0.06, 0.08]:
                    for sl in [-0.06, -0.08]:
                        rules.append({"rule_id": f"trail{act:.0%}_{dd:.0%}_sl{abs(sl):.0%}_{scope}_{mode}", "scope": scope, "exec_mode": mode, "trail_act": act, "trail_dd": dd, "sl": sl})

        for scope in ["all", "D", "D+S"]:
            for tp in [0.20, 0.25]:
                for act in [0.08, 0.12]:
                    for dd in [0.06, 0.08]:
                        rules.append({"rule_id": f"tp{tp:.0%}_trail{act:.0%}_{dd:.0%}_{scope}_{mode}", "scope": scope, "exec_mode": mode, "tp": tp, "trail_act": act, "trail_dd": dd})

        for c_tp in [None, 0.20, 0.25, 0.30]:
            for s_tp in [0.12, 0.15, 0.18, 0.20]:
                for d_tp in [0.12, 0.15, 0.18, 0.20]:
                    label_c = "Craw" if c_tp is None else f"C{c_tp:.0%}"
                    rid = f"line_tp_{label_c}_S{s_tp:.0%}_D{d_tp:.0%}_{mode}"
                    rules.append(
                        {
                            "rule_id": rid,
                            "scope": "all",
                            "exec_mode": mode,
                            "tp_by_line": {"C": c_tp, "S": s_tp, "D": d_tp},
                        }
                    )

    seen = set()
    unique = []
    for rule in rules:
        if rule["rule_id"] not in seen:
            seen.add(rule["rule_id"])
            unique.append(rule)
    return unique


def simulate_one(row, path: dict, rule: dict) -> tuple[pd.Timestamp, float, float, str, int]:
    dates = path["dates"]
    closes = path["closes"]
    if len(closes) == 0:
        return row.exit_date, np.nan, float(row.ret), "missing_price", -1
    entry = float(closes[0])
    original_exit_date = pd.Timestamp(dates[-1])
    original_exit_close = float(closes[-1])

    if rule["rule_id"] == "base_original" or not scope_applies(str(row.line), rule["scope"]):
        return original_exit_date, original_exit_close, original_exit_close / entry - 1.0, "original", len(closes) - 1

    rets = closes / entry - 1.0
    peak = -math.inf
    trigger_idx = None
    reason = "original"

    for i in range(1, len(rets)):
        r = float(rets[i])
        peak = max(peak, r)
        tp = rule.get("tp")
        if "tp_by_line" in rule:
            tp = rule["tp_by_line"].get(str(row.line))
        if "sl" in rule and r <= rule["sl"]:
            trigger_idx, reason = i, f"stop_loss_{rule['sl']:.0%}"
            break
        if tp is not None and r >= tp:
            trigger_idx, reason = i, f"take_profit_{tp:.0%}"
            if "tp_by_line" in rule:
                reason = f"take_profit_{str(row.line)}_{tp:.0%}"
            break
        if "trail_act" in rule and peak >= rule["trail_act"] and r <= peak - rule["trail_dd"]:
            trigger_idx, reason = i, f"trail_{rule['trail_act']:.0%}_{rule['trail_dd']:.0%}"
            break
        if "cap" in rule and i >= rule["cap"]:
            trigger_idx, reason = i, f"cap_{rule['cap']}"
            break

    if trigger_idx is None:
        exit_idx = len(closes) - 1
    elif rule["exec_mode"] == "next_close":
        exit_idx = min(trigger_idx + 1, len(closes) - 1)
    else:
        exit_idx = trigger_idx
    exit_close = float(closes[exit_idx])
    return pd.Timestamp(dates[exit_idx]), exit_close, exit_close / entry - 1.0, reason, int(exit_idx)


def metric_frame(df: pd.DataFrame, ret_col: str = "ret_new") -> dict:
    if df.empty:
        return {"n": 0, "win": 0.0, "avg": 0.0, "sum": 0.0, "pf": 0.0, "annual_avg_sum": 0.0, "worst_year_sum": 0.0}
    r = df[ret_col].astype(float)
    pos = r[r > 0].sum()
    neg = r[r < 0].sum()
    annual = df.groupby("year")[ret_col].sum()
    return {
        "n": int(len(df)),
        "win": float((r > 0).mean()),
        "avg": float(r.mean()),
        "sum": float(r.sum()),
        "pf": float(pos / abs(neg)) if neg < 0 else 99.0,
        "annual_avg_sum": float(annual.mean()),
        "worst_year_sum": float(annual.min()),
    }


def train_score(m: dict, mode: str) -> float:
    pf = min(float(m["pf"]), 8.0)
    if mode == "profit_first":
        return float(m["sum"]) + 1.5 * float(m["worst_year_sum"]) + 0.15 * pf
    if mode == "stable_profit":
        return float(m["sum"]) + 4.0 * float(m["win"]) + 2.0 * float(m["worst_year_sum"]) + 0.25 * pf
    if mode == "win_first":
        return 8.0 * float(m["win"]) + 0.25 * float(m["sum"]) + 0.75 * float(m["worst_year_sum"]) + 0.20 * pf
    raise ValueError(mode)


def strict_select(panel: pd.DataFrame, mode: str, allowed_rule_ids: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if allowed_rule_ids is not None:
        panel = panel[panel["rule_id"].isin(allowed_rule_ids)].copy()
    years = sorted(panel["year"].unique())
    chosen_frames = []
    choices = []
    for y in years:
        train = panel[panel["year"] < y]
        test = panel[panel["year"] == y]
        if train.empty:
            rid, score = "base_original", np.nan
        else:
            rows = []
            for rule_id, g in train.groupby("rule_id"):
                m = metric_frame(g)
                if m["n"] >= 80:
                    rows.append({"rule_id": rule_id, "score": train_score(m, mode), **m})
            ranked = pd.DataFrame(rows).sort_values(["score", "sum", "win"], ascending=False)
            rid = str(ranked.iloc[0]["rule_id"]) if not ranked.empty else "base_original"
            score = float(ranked.iloc[0]["score"]) if not ranked.empty else np.nan
        chosen = test[test["rule_id"].eq(rid)].copy()
        chosen_frames.append(chosen)
        choices.append({"year": int(y), "selector": mode, "chosen_rule": rid, "train_score": score, **metric_frame(chosen)})
    out = pd.concat(chosen_frames, ignore_index=True)
    return out, pd.DataFrame(choices), metric_frame(out)


def selector_rule_sets(rank: pd.DataFrame) -> dict[str, set[str] | None]:
    all_rules = set(rank["rule_id"].astype(str))
    simple_rules = {r for r in all_rules if not r.startswith("line_tp_")}
    tp_only = {
        r
        for r in all_rules
        if r == "base_original" or (r.startswith("tp") and "_sl" not in r and "_trail" not in r)
    }
    conservative_tp = {
        r
        for r in all_rules
        if r == "base_original"
        or r.startswith("tp12%_")
        or r.startswith("tp13%_")
        or r.startswith("tp14%_")
        or r.startswith("tp15%_")
        or r.startswith("tp16%_")
    }
    return {
        "all": None,
        "simple": simple_rules,
        "tp_only": tp_only,
        "conservative_tp": conservative_tp,
    }


def snap_tp15(rule_id: str) -> str:
    for n in ["12", "13", "14", "16"]:
        if rule_id.startswith(f"tp{n}%_"):
            return "tp15%_" + rule_id.split("%_", 1)[1]
    return rule_id


def apply_choices(panel: pd.DataFrame, choices: pd.DataFrame, selector_name: str, snap_center15: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    selected = []
    fixed_choices = []
    for row in choices.itertuples(index=False):
        rid = str(row.chosen_rule)
        chosen = snap_tp15(rid) if snap_center15 else rid
        if not panel["rule_id"].eq(chosen).any():
            chosen = rid
        part = panel[(panel["year"].eq(int(row.year))) & (panel["rule_id"].eq(chosen))].copy()
        part["selector"] = selector_name
        selected.append(part)
        fixed_choices.append(
            {
                "year": int(row.year),
                "selector": selector_name,
                "raw_chosen_rule": rid,
                "chosen_rule": chosen,
                "train_score": row.train_score,
                **metric_frame(part),
            }
        )
    out = pd.concat(selected, ignore_index=True)
    choices_out = pd.DataFrame(fixed_choices)
    return out, choices_out, metric_frame(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trades = load_v2_trades()
    paths, sources = build_paths(trades)
    sources.to_csv(OUT_DIR / "price_sources.csv", index=False, encoding="utf-8-sig")
    rules = make_rules()
    print(f"trades={len(trades)}, rules={len(rules)}")

    rows = []
    trade_rows = list(trades.itertuples(index=False))
    for idx, rule in enumerate(rules, 1):
        if idx % 80 == 0:
            print(f"simulated {idx}/{len(rules)}")
        for row in trade_rows:
            exit_date, exit_close, ret_new, reason, hold_bars = simulate_one(row, paths[int(row.trade_id)], rule)
            rows.append(
                {
                    "rule_id": rule["rule_id"],
                    "scope": rule.get("scope", "all"),
                    "exec_mode": rule.get("exec_mode", "original"),
                    "trade_id": int(row.trade_id),
                    "year": int(row.year),
                    "line": row.line,
                    "symbol": row.symbol,
                    "name": row.name,
                    "signal_date": row.signal_date,
                    "entry_date": row.entry_date,
                    "original_exit_date": row.exit_date,
                    "exit_date_new": exit_date,
                    "exit_close_new": exit_close,
                    "ret_original": float(row.ret),
                    "ret_new": ret_new,
                    "exit_reason": reason,
                    "hold_bars": hold_bars,
                    "policy": row.policy,
                }
            )

    panel = pd.DataFrame(rows)
    panel.to_csv(OUT_DIR / "exit_candidate_panel.csv", index=False, encoding="utf-8-sig")

    ranks = []
    for rid, g in panel.groupby("rule_id"):
        first = g.iloc[0]
        ranks.append({"rule_id": rid, "scope": first["scope"], "exec_mode": first["exec_mode"], **metric_frame(g)})
    rank = pd.DataFrame(ranks).sort_values(["sum", "win", "pf"], ascending=False)
    rank.to_csv(OUT_DIR / "full_sample_exit_rank.csv", index=False, encoding="utf-8-sig")

    summaries = []
    saved_choices: dict[str, pd.DataFrame] = {}
    rule_sets = selector_rule_sets(rank)
    for family, allowed in rule_sets.items():
        for mode in ["profit_first", "stable_profit", "win_first"]:
            selector_name = f"{family}_{mode}"
            selected, choices, m = strict_select(panel, mode, allowed)
            choices["selector_family"] = family
            choices["score_mode"] = mode
            selected["selector_family"] = family
            selected["score_mode"] = mode
            selected.to_csv(OUT_DIR / f"strict_rolling_{selector_name}_trades.csv", index=False, encoding="utf-8-sig")
            choices.to_csv(OUT_DIR / f"strict_rolling_{selector_name}_choices.csv", index=False, encoding="utf-8-sig")
            saved_choices[selector_name] = choices
            summaries.append({"selector": selector_name, **m})
    if "conservative_tp_profit_first" in saved_choices:
        selector_name = "conservative_tp_center15_profit_first"
        selected, choices, m = apply_choices(panel, saved_choices["conservative_tp_profit_first"], selector_name, snap_center15=True)
        selected.to_csv(OUT_DIR / f"strict_rolling_{selector_name}_trades.csv", index=False, encoding="utf-8-sig")
        choices.to_csv(OUT_DIR / f"strict_rolling_{selector_name}_choices.csv", index=False, encoding="utf-8-sig")
        summaries.append({"selector": selector_name, **m})
    summary = pd.DataFrame(summaries).sort_values(["sum", "win", "pf"], ascending=False)
    summary.to_csv(OUT_DIR / "strict_rolling_exit_summary.csv", index=False, encoding="utf-8-sig")

    baseline = rank[rank["rule_id"].eq("base_original")].iloc[0].to_dict()
    best_full = rank.iloc[0].to_dict()
    best_roll = summary.iloc[0].to_dict()
    report = [
        "# C/S/D v2 exit-layer test",
        "",
        "Only the exit layer is changed. Entry signals, C/S/D judge, ETF universe, and v2 selected trades are fixed.",
        "",
        "All stop/take-profit/trailing rules are close-price based. `same_close` is a near-close execution approximation; `next_close` is the stricter next-trading-day close version.",
        "",
        "## Baseline v2",
        f"- n: {int(baseline['n'])}",
        f"- win: {baseline['win']:.2%}",
        f"- avg trade: {baseline['avg']:.2%}",
        f"- return sum: {baseline['sum']:.2%}",
        f"- profit factor: {baseline['pf']:.2f}",
        f"- annual avg sum: {baseline['annual_avg_sum']:.2%}",
        f"- worst year sum: {baseline['worst_year_sum']:.2%}",
        "",
        "## Full-sample diagnostic best",
        f"- rule: {best_full['rule_id']}",
        f"- win: {best_full['win']:.2%}",
        f"- avg trade: {best_full['avg']:.2%}",
        f"- return sum: {best_full['sum']:.2%}",
        f"- profit factor: {best_full['pf']:.2f}",
        f"- annual avg sum: {best_full['annual_avg_sum']:.2%}",
        f"- worst year sum: {best_full['worst_year_sum']:.2%}",
        "",
        "## Strict rolling best",
        f"- selector: {best_roll['selector']}",
        f"- win: {best_roll['win']:.2%}",
        f"- avg trade: {best_roll['avg']:.2%}",
        f"- return sum: {best_roll['sum']:.2%}",
        f"- profit factor: {best_roll['pf']:.2f}",
        f"- annual avg sum: {best_roll['annual_avg_sum']:.2%}",
        f"- worst year sum: {best_roll['worst_year_sum']:.2%}",
    ]
    (OUT_DIR / "exit_guard_report.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False))
    print("baseline:", baseline)
    print("best_full:", best_full)


if __name__ == "__main__":
    main()
