from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
D_OUT = ROOT / "D" / "out"
EXIT_DIR = D_OUT / "csd_v2_exit_guard"
BASE_DIR = D_OUT / "csd_regime_judge_v2_strict"
FORMAL_DIR = ROOT / "D" / "formal"
BEST_SELECTOR = "all_priority_bias_min3"
BEST_EXIT_SELECTOR = "conservative_tp_center15_profit_first"


def profit_factor(ret: pd.Series) -> float:
    pos = ret[ret > 0].sum()
    neg = -ret[ret < 0].sum()
    if neg == 0:
        return 99.0 if pos > 0 else 0.0
    return float(pos / neg)


def metrics(df: pd.DataFrame, ret_col: str = "ret_eval") -> dict:
    if df.empty:
        return {
            "n_total": 0,
            "n_active": 0,
            "active_rate": 0.0,
            "win": 0.0,
            "avg": 0.0,
            "sum": 0.0,
            "pf": 0.0,
            "annual_avg_sum": 0.0,
            "worst_year_sum": 0.0,
            "positive_year_rate": 0.0,
        }
    active = df[df["active"]].copy()
    r_all = df[ret_col].fillna(0.0).astype(float)
    r_active = active[ret_col].astype(float)
    annual = df.groupby("year")[ret_col].sum()
    return {
        "n_total": int(len(df)),
        "n_active": int(len(active)),
        "active_rate": float(len(active) / len(df)),
        "win": float((r_active > 0).mean()) if len(active) else 0.0,
        "avg": float(r_active.mean()) if len(active) else 0.0,
        "sum": float(r_all.sum()),
        "pf": profit_factor(r_active) if len(active) else 0.0,
        "annual_avg_sum": float(annual.mean()) if len(annual) else 0.0,
        "worst_year_sum": float(annual.min()) if len(annual) else 0.0,
        "positive_year_rate": float((annual > 0).mean()) if len(annual) else 0.0,
    }


def load_joined() -> pd.DataFrame:
    trades = pd.read_csv(
        EXIT_DIR / f"strict_rolling_{BEST_EXIT_SELECTOR}_trades.csv",
        parse_dates=["signal_date", "entry_date", "original_exit_date", "exit_date_new"],
        dtype={"symbol": str},
    )
    base = pd.read_csv(
        BASE_DIR / "extra_selector_selected_trades.csv",
        parse_dates=["signal_date", "entry_date", "exit_date"],
        dtype={"symbol": str},
        low_memory=False,
    )
    base = base[base["selector"].eq(BEST_SELECTOR)].copy().reset_index(drop=True)
    base["trade_id"] = np.arange(len(base))
    keep_cols = [
        "trade_id",
        "rank_score",
        "trend_score",
        "rebound_score",
        "panic_score",
        "median_ret20",
        "median_ret60",
        "low_pos_rate",
        "deep_drop_rate",
        "median_dd60",
        "sym_252_n",
        "sym_252_avg",
        "sym_252_win",
        "sym_252_pf",
        "line_252_n",
        "line_252_avg",
        "line_252_win",
        "line_252_pf",
        "sym_504_n",
        "sym_504_avg",
        "sym_504_win",
        "sym_504_pf",
        "line_504_n",
        "line_504_avg",
        "line_504_win",
        "line_504_pf",
        "selected_param_id",
        "cold_start",
    ]
    out = trades.merge(base[keep_cols], on="trade_id", how="left")
    out["ret_eval"] = out["ret_new"].astype(float)
    out["active"] = True
    return out


def apply_rule(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = df.copy()
    keep = pd.Series(True, index=out.index)
    if rule == "base_exit_guard":
        pass
    elif rule.startswith("rank_ge_"):
        keep &= out["rank_score"].fillna(0) >= float(rule.split("_")[-1])
    elif rule.startswith("cool_symbol_"):
        days = int(rule.split("_")[-1])
        last: dict[str, pd.Timestamp] = {}
        marks = []
        for row in out.sort_values(["signal_date", "trade_id"]).itertuples():
            sym = str(row.symbol)
            d = pd.Timestamp(row.signal_date)
            ok = sym not in last or (d - last[sym]).days > days
            marks.append((row.Index, ok))
            if ok:
                last[sym] = d
        keep &= pd.Series({idx: ok for idx, ok in marks})
    elif rule.startswith("cool_line_symbol_"):
        days = int(rule.split("_")[-1])
        last: dict[tuple[str, str], pd.Timestamp] = {}
        marks = []
        for row in out.sort_values(["signal_date", "trade_id"]).itertuples():
            key = (str(row.line), str(row.symbol))
            d = pd.Timestamp(row.signal_date)
            ok = key not in last or (d - last[key]).days > days
            marks.append((row.Index, ok))
            if ok:
                last[key] = d
        keep &= pd.Series({idx: ok for idx, ok in marks})
    elif rule.startswith("no_line_overheat_"):
        _, _, _, t, r, m = rule.split("_")
        keep &= ~(
            (out["trend_score"].fillna(0) > float(t))
            & (out["rebound_score"].fillna(1) < float(r))
            & (out["median_ret20"].fillna(-1) > float(m))
        )
    elif rule.startswith("no_s_overheat_"):
        _, _, _, t, r, m = rule.split("_")
        keep &= ~(
            (out["line"].eq("S"))
            & (out["trend_score"].fillna(0) > float(t))
            & (out["rebound_score"].fillna(1) < float(r))
            & (out["median_ret20"].fillna(-1) > float(m))
        )
    elif rule.startswith("history_sym_win_ge_"):
        th = float(rule.split("_")[-1])
        keep &= out["sym_504_win"].fillna(1.0) >= th
    elif rule.startswith("history_line_win_ge_"):
        th = float(rule.split("_")[-1])
        keep &= out["line_504_win"].fillna(1.0) >= th
    elif rule.startswith("need_dd60_le_"):
        th = float(rule.split("_")[-1])
        keep &= out["median_dd60"].fillna(-1.0) <= th
    else:
        raise ValueError(rule)

    out["active"] = keep.fillna(False)
    out["ret_eval"] = np.where(out["active"], out["ret_new"].astype(float), 0.0)
    out["extra_rule"] = rule
    return out


def candidate_rules() -> list[str]:
    rules = ["base_exit_guard"]
    rules += [f"rank_ge_{x}" for x in [25, 35, 45, 55, 65, 75]]
    rules += [f"cool_symbol_{x}" for x in [2, 3, 5, 7, 10]]
    rules += [f"cool_line_symbol_{x}" for x in [2, 3, 5, 7, 10]]
    for t in [0.45, 0.55, 0.65, 0.75]:
        for r in [0.20, 0.30, 0.40]:
            for m in [-0.02, 0.00, 0.03, 0.06]:
                rules.append(f"no_line_overheat_{t}_{r}_{m}")
                rules.append(f"no_s_overheat_{t}_{r}_{m}")
    rules += [f"history_sym_win_ge_{x}" for x in [0.42, 0.46, 0.50, 0.54]]
    rules += [f"history_line_win_ge_{x}" for x in [0.42, 0.46, 0.50, 0.54]]
    rules += [f"need_dd60_le_{x}" for x in [-0.02, -0.04, -0.06, -0.08, -0.10]]
    return rules


def score(m: dict, base_active_rate: float) -> float:
    # Keep frequency meaningful; rules under 90% activity are diagnostic only.
    trigger_penalty = max(0.0, 0.90 * base_active_rate - m["active_rate"]) * 35.0
    return (
        m["sum"]
        + 2.0 * m["worst_year_sum"]
        + 3.0 * m["positive_year_rate"]
        + 5.0 * m["win"]
        + min(m["pf"], 6.0) * 0.5
        - trigger_penalty
    )


def rolling_pick(df: pd.DataFrame, rules: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = {rule: apply_rule(df, rule) for rule in rules}
    years = sorted(df["year"].unique())
    base_rate = metrics(frames["base_exit_guard"])["active_rate"]
    selected_parts = []
    choices = []
    ranks = []
    for y in years:
        train_years = [yy for yy in years if yy < y]
        if len(train_years) < 3:
            chosen = "base_exit_guard"
            train_score = np.nan
        else:
            rows = []
            for rule, frame in frames.items():
                tr = frame[frame["year"] < y]
                m = metrics(tr)
                rows.append({"year": y, "rule": rule, "score": score(m, base_rate), **m})
            rank = pd.DataFrame(rows).sort_values(["score", "sum", "win"], ascending=False)
            ranks.append(rank)
            chosen = str(rank.iloc[0]["rule"])
            train_score = float(rank.iloc[0]["score"])
        part = frames[chosen][frames[chosen]["year"].eq(y)].copy()
        selected_parts.append(part)
        choices.append({"year": int(y), "chosen_rule": chosen, "train_score": train_score, **metrics(part)})
    selected = pd.concat(selected_parts, ignore_index=True)
    choices_df = pd.DataFrame(choices)
    ranks_df = pd.concat(ranks, ignore_index=True) if ranks else pd.DataFrame()
    return selected, choices_df, ranks_df


def main() -> None:
    FORMAL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_joined()
    rules = candidate_rules()
    fixed_rows = []
    fixed_frames = []
    for rule in rules:
        fr = apply_rule(df, rule)
        fixed_rows.append({"rule": rule, **metrics(fr)})
        fixed_frames.append(fr)
    fixed_rank = pd.DataFrame(fixed_rows).sort_values(["sum", "active_rate", "win"], ascending=False)
    fixed_rank.to_csv(FORMAL_DIR / "extra_overlay_fixed_rank.csv", index=False, encoding="utf-8-sig")

    selected, choices, train_ranks = rolling_pick(df, rules)
    selected.to_csv(FORMAL_DIR / "formal_selected_trades.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(FORMAL_DIR / "formal_overlay_choices.csv", index=False, encoding="utf-8-sig")
    train_ranks.to_csv(FORMAL_DIR / "formal_overlay_train_ranks.csv", index=False, encoding="utf-8-sig")

    base = apply_rule(df, "base_exit_guard")
    compare = pd.DataFrame(
        [
            {"version": "formal_base_exit_guard", **metrics(base)},
            {"version": "strict_rolling_extra_overlay", **metrics(selected)},
        ]
    )
    compare.to_csv(FORMAL_DIR / "formal_compare.csv", index=False, encoding="utf-8-sig")

    annual = pd.concat(
        [
            base.assign(version="formal_base_exit_guard"),
            selected.assign(version="strict_rolling_extra_overlay"),
        ],
        ignore_index=True,
    )
    annual_summary = (
        annual.groupby(["version", "year"])
        .agg(
            trades=("trade_id", "count"),
            active=("active", "sum"),
            win=("ret_eval", lambda s: float((s[s != 0] > 0).mean()) if (s != 0).any() else 0.0),
            avg=("ret_eval", "mean"),
            sum=("ret_eval", "sum"),
        )
        .reset_index()
    )
    annual_summary.to_csv(FORMAL_DIR / "formal_annual_compare.csv", index=False, encoding="utf-8-sig")

    md = [
        "# Formal C/S/D strict rolling upgrade",
        "",
        "Current formal candidate:",
        "",
        "`all_priority_bias_min3 + conservative_tp_center15_profit_first`",
        "",
        "This keeps the existing C/S/D entries and adds a strict rolling exit layer. The useful rule selected by past years is usually about 15% take-profit, executed at the next close.",
        "",
        "## Extra overlay exploration",
        "",
        "I also tested score thresholds, cooldowns, overheat filters, historical win-rate filters, and drawdown gates on top of the exit guard. The formal overlay selector is itself rolling: every test year only sees earlier years.",
        "",
        "Result: the extra overlay is kept as an audit candidate only unless it beats the base exit guard without reducing trigger rate too much.",
        "",
        compare.to_markdown(index=False),
    ]
    (FORMAL_DIR / "FORMAL_STRATEGY.md").write_text("\n".join(md), encoding="utf-8")
    print(compare.to_string(index=False))
    print("top fixed overlays:")
    print(fixed_rank.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
