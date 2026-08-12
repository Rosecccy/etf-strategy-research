from __future__ import annotations

from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRADES_PATH = ROOT / "D" / "out" / "line_specific_gates" / "rolling_line_gates_trades.csv"
OUT_DIR = ROOT / "D" / "out" / "csd_regime_judge_v1"


def pf(s: pd.Series) -> float:
    w = s[s > 0].sum()
    l = -s[s < 0].sum()
    if l == 0:
        return 99.0 if w > 0 else 0.0
    return float(w / l)


def summarize(df: pd.DataFrame, ret_col: str = "ret") -> dict[str, float]:
    x = df[df[ret_col].notna()].copy()
    r = x[ret_col]
    annual = x.groupby("year")[ret_col].sum() if len(x) else pd.Series(dtype=float)
    return {
        "n": int(len(x)),
        "win": float((r > 0).mean()) if len(x) else np.nan,
        "avg": float(r.mean()) if len(x) else np.nan,
        "sum": float(r.sum()) if len(x) else 0.0,
        "pf": pf(r) if len(x) else np.nan,
        "annual_avg_sum": float(annual.mean()) if len(annual) else np.nan,
        "worst_year_sum": float(annual.min()) if len(annual) else np.nan,
    }


def policy_score(g: pd.DataFrame) -> float:
    r = g["ret"].dropna()
    if len(r) < 30:
        return -1e9
    yearly = g.groupby("year")["ret"].sum()
    return float(r.sum() + (r > 0).mean() * 8.0 + min(pf(r), 8.0) + min(yearly.min(), 0.0) * 1.5)


def prepare_active() -> pd.DataFrame:
    df = pd.read_csv(TRADES_PATH, parse_dates=["signal_date", "entry_date", "exit_date"])
    df = df[(df["active"] == True) & (df["completed"] == True)].copy()
    df["year"] = df["signal_date"].dt.year
    # 如果同一天同一线有多个候选，只保留 rank_score 最高的一个。
    df = df.sort_values(["signal_date", "line", "rank_score"], ascending=[True, True, False])
    df = df.drop_duplicates(["signal_date", "line"], keep="first")
    return df


def choose_by_priority(day: pd.DataFrame, order: tuple[str, ...]) -> pd.Series:
    for line in order:
        hit = day[day["line"] == line]
        if len(hit):
            return hit.iloc[0]
    return day.iloc[0]


def make_policy_panel(active: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = list(active.groupby("signal_date", sort=True))

    def add(policy: str, row: pd.Series) -> None:
        d = row.to_dict()
        d["policy"] = policy
        rows.append(d)

    for date, day in groups:
        lines = set(day["line"])
        state = day.iloc[0]

        # 基础固定优先级。
        for order in permutations(("C", "S", "D")):
            if not (lines & set(order)):
                continue
            add("priority_" + "".join(order), choose_by_priority(day, order))

        # 固定偏好。
        for line in ["C", "S", "D"]:
            hit = day[day["line"] == line]
            if len(hit):
                add(f"only_{line}_if_available", hit.iloc[0])
            else:
                add(f"only_{line}_else_DCS", choose_by_priority(day, ("D", "C", "S")))

        # 状态裁判：C=极端反弹，S=趋势补充，D=默认主力。
        for c_trend in [0.45, 0.55]:
            for c_rebound in [0.55, 0.65]:
                for s_trend in [0.45, 0.60]:
                    for s_rebound in [0.30, 0.45]:
                        c_ok = (
                            ("C" in lines)
                            and (float(state["trend_score"]) <= c_trend)
                            and (float(state["rebound_score"]) >= c_rebound)
                        )
                        s_ok = (
                            ("S" in lines)
                            and (float(state["trend_score"]) >= s_trend)
                            and (float(state["rebound_score"]) <= s_rebound)
                        )
                        if c_ok:
                            chosen = day[day["line"] == "C"].iloc[0]
                        elif s_ok:
                            chosen = day[day["line"] == "S"].iloc[0]
                        else:
                            chosen = choose_by_priority(day, ("D", "C", "S"))
                        add(f"state_Ct{c_trend:.2f}_Cr{c_rebound:.2f}_St{s_trend:.2f}_Sr{s_rebound:.2f}", chosen)

        # 恐慌裁判：恐慌高且反弹高才允许 C 抢反弹；趋势强且恐慌低用 S；否则 D。
        for panic_hi in [0.70, 0.85]:
            for panic_lo in [0.35, 0.50]:
                for trend_hi in [0.55, 0.70]:
                    c_ok = (
                        ("C" in lines)
                        and (float(state["panic_score"]) >= panic_hi)
                        and (float(state["rebound_score"]) >= 0.55)
                    )
                    s_ok = (
                        ("S" in lines)
                        and (float(state["trend_score"]) >= trend_hi)
                        and (float(state["panic_score"]) <= panic_lo)
                    )
                    if c_ok:
                        chosen = day[day["line"] == "C"].iloc[0]
                    elif s_ok:
                        chosen = day[day["line"] == "S"].iloc[0]
                    else:
                        chosen = choose_by_priority(day, ("D", "C", "S"))
                    add(f"panic_Ph{panic_hi:.2f}_Pl{panic_lo:.2f}_Th{trend_hi:.2f}", chosen)

        # 质量裁判：用滚动历史质量字段，但仍只在当日候选内选。
        for lookback in [252, 504, 99999]:
            avg_col = f"line_{lookback}_avg"
            win_col = f"line_{lookback}_win"
            pf_col = f"line_{lookback}_pf"
            if avg_col in day.columns:
                tmp = day.copy()
                tmp["_q"] = tmp[avg_col].fillna(-9) * 100 + tmp[win_col].fillna(0) * 5 + np.log1p(tmp[pf_col].clip(lower=0).fillna(0))
                add(f"quality_line_lb{lookback}", tmp.sort_values("_q", ascending=False).iloc[0])
            avg_col = f"sym_{lookback}_avg"
            win_col = f"sym_{lookback}_win"
            pf_col = f"sym_{lookback}_pf"
            if avg_col in day.columns:
                tmp = day.copy()
                tmp["_q"] = tmp[avg_col].fillna(-9) * 100 + tmp[win_col].fillna(0) * 5 + np.log1p(tmp[pf_col].clip(lower=0).fillna(0))
                add(f"quality_symbol_lb{lookback}", tmp.sort_values("_q", ascending=False).iloc[0])

    panel = pd.DataFrame(rows)
    if "date" in panel.columns:
        panel = panel.drop(columns=["date"])
    panel["year"] = pd.to_datetime(panel["signal_date"]).dt.year
    return panel


def select_rolling(panel: pd.DataFrame, min_train_years: int, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = []
    choices = []
    years = sorted(panel["year"].unique())
    for year in years:
        test = panel[panel["year"] == year]
        train_years = [y for y in years if y < year]
        if len(train_years) < min_train_years:
            chosen = "priority_DCS"
            rank = pd.DataFrame()
        else:
            train = panel[panel["year"] < year]
            rows = []
            for p, g in train.groupby("policy"):
                s = summarize(g)
                s["policy"] = p
                s["score"] = policy_score(g)
                rows.append(s)
            rank = pd.DataFrame(rows).sort_values(["score", "sum"], ascending=False)
            base = rank[rank["policy"] == "priority_DCS"].iloc[0]
            if mode == "argmax":
                chosen = str(rank.iloc[0]["policy"])
            elif mode == "beat_base":
                eligible = rank[
                    (rank["sum"] > float(base["sum"]))
                    & (rank["pf"] >= float(base["pf"]) * 0.98)
                    & (rank["worst_year_sum"] >= float(base["worst_year_sum"]) - 0.5)
                ]
                chosen = str(eligible.iloc[0]["policy"]) if len(eligible) else "priority_DCS"
            elif mode == "stable":
                eligible = rank[
                    (rank["sum"] > float(base["sum"]) + 1.0)
                    & (rank["win"] >= float(base["win"]))
                    & (rank["worst_year_sum"] >= float(base["worst_year_sum"]))
                ]
                chosen = str(eligible.iloc[0]["policy"]) if len(eligible) else "priority_DCS"
            else:
                raise ValueError(mode)
        picked = test[test["policy"] == chosen].copy()
        picked["selector_mode"] = mode
        picked["min_train_years"] = min_train_years
        selected.append(picked)
        choices.append(
            {
                "year": int(year),
                "mode": mode,
                "min_train_years": min_train_years,
                "chosen_policy": chosen,
                "test_n": int(len(picked)),
                "test_win": float((picked["ret"] > 0).mean()) if len(picked) else np.nan,
                "test_sum": float(picked["ret"].sum()) if len(picked) else 0.0,
                "top_policy_in_train": "" if rank.empty else str(rank.iloc[0]["policy"]),
            }
        )
    return pd.concat(selected, ignore_index=True), pd.DataFrame(choices)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    active = prepare_active()
    active.to_csv(OUT_DIR / "active_line_candidates.csv", index=False, encoding="utf-8-sig")

    panel = make_policy_panel(active)
    panel.to_csv(OUT_DIR / "policy_candidate_panel.csv", index=False, encoding="utf-8-sig")

    full_rows = []
    for p, g in panel.groupby("policy"):
        s = summarize(g)
        s["policy"] = p
        full_rows.append(s)
    full_rank = pd.DataFrame(full_rows).sort_values(["sum", "win"], ascending=False)
    full_rank.to_csv(OUT_DIR / "full_sample_policy_rank.csv", index=False, encoding="utf-8-sig")

    selected_all = []
    choices_all = []
    for min_train in [1, 2, 3, 5]:
        for mode in ["argmax", "beat_base", "stable"]:
            selected, choices = select_rolling(panel, min_train, mode)
            selected_all.append(selected)
            choices_all.append(choices)
    selected_df = pd.concat(selected_all, ignore_index=True)
    choices_df = pd.concat(choices_all, ignore_index=True)
    selected_df.to_csv(OUT_DIR / "strict_rolling_selected_trades.csv", index=False, encoding="utf-8-sig")
    choices_df.to_csv(OUT_DIR / "strict_rolling_choices.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for p in ["priority_DCS", "priority_CSD", "priority_SDC", "quality_line_lb252", "quality_symbol_lb252"]:
        if p in set(panel["policy"]):
            summary_rows.append({"model": f"fixed_{p}", **summarize(panel[panel["policy"] == p])})
    for (mode, min_train), g in selected_df.groupby(["selector_mode", "min_train_years"]):
        summary_rows.append({"model": f"rolling_{mode}_mintrain{min_train}", **summarize(g)})
    summary = pd.DataFrame(summary_rows).sort_values(["sum", "win"], ascending=False)
    summary.to_csv(OUT_DIR / "strict_rolling_summary.csv", index=False, encoding="utf-8-sig")

    annual = []
    for model_name, g in [("fixed_priority_DCS", panel[panel["policy"] == "priority_DCS"])]:
        y = g.groupby("year").agg(n=("ret", "size"), win=("ret", lambda s: (s > 0).mean()), avg=("ret", "mean"), sum=("ret", "sum")).reset_index()
        y.insert(0, "model", model_name)
        annual.append(y)
    best_model = summary.iloc[0]["model"]
    if str(best_model).startswith("rolling_"):
        parts = str(best_model).replace("rolling_", "").split("_mintrain")
        mode = parts[0]
        min_train = int(parts[1])
        g = selected_df[(selected_df["selector_mode"] == mode) & (selected_df["min_train_years"] == min_train)]
        y = g.groupby("year").agg(n=("ret", "size"), win=("ret", lambda s: (s > 0).mean()), avg=("ret", "mean"), sum=("ret", "sum")).reset_index()
        y.insert(0, "model", str(best_model))
        annual.append(y)
    annual_df = pd.concat(annual, ignore_index=True)
    annual_df.to_csv(OUT_DIR / "best_vs_base_annual.csv", index=False, encoding="utf-8-sig")

    print("SUMMARY")
    print(summary.to_string(index=False))
    print("\nTOP FULL SAMPLE POLICIES")
    print(full_rank.head(20).to_string(index=False))
    print("\nCHOICES")
    print(choices_df.to_string(index=False))


if __name__ == "__main__":
    main()
