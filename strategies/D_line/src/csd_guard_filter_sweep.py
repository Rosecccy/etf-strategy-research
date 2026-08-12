from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "D" / "out" / "csd_regime_judge_v2_strict" / "policy_candidate_panel.csv"
OUT = ROOT / "D" / "out" / "csd_guard_filter_sweep"
BASE_POLICY = "priority_CSD_g40_t055_r030_noskip"


@dataclass(frozen=True)
class Filter:
    key: str
    label: str


def pf(ret: pd.Series) -> float:
    win = ret[ret > 0].sum()
    loss = -ret[ret < 0].sum()
    if loss == 0:
        return 99.0 if win > 0 else 0.0
    return float(win / loss)


def summarize(df: pd.DataFrame, ret_col: str = "ret2") -> dict[str, float]:
    r = df[ret_col].fillna(0.0).astype(float)
    active = df[df["keep"]].copy()
    ar = active[ret_col].astype(float)
    annual = df.groupby(df["signal_date"].dt.year)[ret_col].sum()
    return {
        "n_total": int(len(df)),
        "n_active": int(len(active)),
        "active_rate": float(len(active) / len(df)) if len(df) else 0.0,
        "win": float((ar > 0).mean()) if len(ar) else np.nan,
        "avg": float(ar.mean()) if len(ar) else np.nan,
        "sum": float(r.sum()),
        "pf": pf(ar) if len(ar) else np.nan,
        "annual_avg_sum": float(annual.mean()) if len(annual) else np.nan,
        "worst_year_sum": float(annual.min()) if len(annual) else np.nan,
        "positive_year_rate": float((annual > 0).mean()) if len(annual) else np.nan,
    }


def score(st: dict, base: dict) -> float:
    # Training-only score. Return is still first, but 2026-style damage is
    # controlled through worst-year and trigger-rate terms.
    active_penalty = max(0.0, base["active_rate"] * 0.70 - st["active_rate"]) * 12.0
    return float(
        st["sum"]
        + st["win"] * 8.0
        + min(st["pf"], 8.0)
        + st["positive_year_rate"] * 2.0
        + min(st["worst_year_sum"], 0.0) * 4.0
        - active_penalty
    )


def read_base() -> pd.DataFrame:
    usecols = [
        "signal_date", "line", "symbol", "name", "rank_score", "ret", "year",
        "trend_score", "rebound_score", "panic_score", "median_ret20",
        "median_ret60", "low_pos_rate", "deep_drop_rate", "median_dd60",
        "sym_252_n", "sym_252_avg", "sym_252_win", "sym_252_pf",
        "line_252_n", "line_252_avg", "line_252_win", "line_252_pf",
        "sym_504_n", "sym_504_avg", "sym_504_win", "sym_504_pf",
        "line_504_n", "line_504_avg", "line_504_win", "line_504_pf",
        "policy",
    ]
    df = pd.read_csv(PANEL, usecols=usecols, parse_dates=["signal_date"], low_memory=False)
    df = df[df["policy"] == BASE_POLICY].copy()
    df = df.sort_values("signal_date").reset_index(drop=True)
    for col in df.columns:
        if col not in ("signal_date", "line", "symbol", "name", "policy"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def apply_filter(df: pd.DataFrame, flt: Filter) -> pd.DataFrame:
    out = df.copy()
    out["keep"] = True
    key = flt.key

    if "__" in key:
        left, right = key.split("__", 1)
        a = apply_filter(df, Filter(left, left))
        b = apply_filter(df, Filter(right, right))
        out["keep"] = a["keep"] & b["keep"]
    elif key == "base":
        pass
    elif key.startswith("rank_ge_"):
        th = float(key.split("_")[-1])
        out["keep"] = out["rank_score"] >= th
    elif key.startswith("weak_cash_"):
        _, _, t, r, p = key.split("_")
        out["keep"] = ~(
            (out["trend_score"] < float(t))
            & (out["rebound_score"] < float(r))
            & (out["panic_score"] < float(p))
        )
    elif key.startswith("no_d_overheat_"):
        _, _, _, t, r, m = key.split("_")
        out["keep"] = ~(
            (out["line"] == "D")
            & (out["trend_score"] > float(t))
            & (out["rebound_score"] < float(r))
            & (out["median_ret20"] > float(m))
        )
    elif key.startswith("no_s_overheat_"):
        _, _, _, t, r, m = key.split("_")
        out["keep"] = ~(
            (out["line"] == "S")
            & (out["trend_score"] > float(t))
            & (out["rebound_score"] < float(r))
            & (out["median_ret20"] > float(m))
        )
    elif key.startswith("cool_symbol_"):
        days = int(key.split("_")[-1])
        keep = []
        last_seen: dict[str, pd.Timestamp] = {}
        for _, row in out.iterrows():
            symbol = str(row["symbol"])
            day = pd.Timestamp(row["signal_date"])
            last = last_seen.get(symbol)
            ok = last is None or (day - last).days > days
            keep.append(ok)
            if ok:
                last_seen[symbol] = day
        out["keep"] = keep
    elif key.startswith("cool_line_symbol_"):
        days = int(key.split("_")[-1])
        keep = []
        last_seen: dict[tuple[str, str], pd.Timestamp] = {}
        for _, row in out.iterrows():
            code = (str(row["line"]), str(row["symbol"]))
            day = pd.Timestamp(row["signal_date"])
            last = last_seen.get(code)
            ok = last is None or (day - last).days > days
            keep.append(ok)
            if ok:
                last_seen[code] = day
        out["keep"] = keep
    elif key.startswith("quality_sym_win_ge_"):
        th = float(key.split("_")[-1])
        out["keep"] = out["sym_252_win"].fillna(1.0) >= th
    elif key.startswith("quality_line_win_ge_"):
        th = float(key.split("_")[-1])
        out["keep"] = out["line_252_win"].fillna(1.0) >= th
    elif key.startswith("quality_sym_pf_ge_"):
        th = float(key.split("_")[-1])
        out["keep"] = out["sym_252_pf"].fillna(99.0) >= th
    elif key.startswith("avoid_high_lowpos_"):
        th = float(key.split("_")[-1])
        out["keep"] = out["low_pos_rate"].fillna(0.0) <= th
    elif key.startswith("need_drawdown_"):
        th = float(key.split("_")[-1])
        out["keep"] = out["median_dd60"].fillna(-1.0) <= th
    else:
        raise ValueError(key)

    out["ret2"] = np.where(out["keep"], out["ret"], 0.0)
    out["filter_key"] = key
    out["filter_label"] = flt.label
    return out


def filters() -> list[Filter]:
    fs = [Filter("base", "不加过滤")]
    fs += [Filter(f"rank_ge_{x}", f"分数不低于 {x}") for x in (20, 35, 50, 65, 80)]
    for t in (0.35, 0.45, 0.55):
        for r in (0.25, 0.35, 0.45):
            for p in (0.45, 0.55, 0.65):
                fs.append(Filter(f"weak_cash_{t}_{r}_{p}", f"弱势空仓 t<{t}, r<{r}, p<{p}"))
    for t in (0.45, 0.55, 0.65):
        for r in (0.25, 0.35, 0.45):
            for m in (-0.02, 0.00, 0.02, 0.05):
                fs.append(Filter(f"no_d_overheat_{t}_{r}_{m}", f"D线过热过滤 t>{t}, r<{r}, m20>{m}"))
                fs.append(Filter(f"no_s_overheat_{t}_{r}_{m}", f"S线过热过滤 t>{t}, r<{r}, m20>{m}"))
    fs += [Filter(f"cool_symbol_{d}", f"同ETF冷却 {d}日") for d in (3, 5, 7, 10, 15, 20)]
    fs += [Filter(f"cool_line_symbol_{d}", f"同线同ETF冷却 {d}日") for d in (3, 5, 7, 10, 15, 20)]
    fs += [Filter(f"quality_sym_win_ge_{x}", f"ETF历史胜率 >= {x}") for x in (0.42, 0.46, 0.50, 0.54)]
    fs += [Filter(f"quality_line_win_ge_{x}", f"线历史胜率 >= {x}") for x in (0.42, 0.46, 0.50, 0.54)]
    fs += [Filter(f"quality_sym_pf_ge_{x}", f"ETF历史PF >= {x}") for x in (0.8, 1.0, 1.2, 1.5)]
    fs += [Filter(f"avoid_high_lowpos_{x}", f"低位占比 <= {x}") for x in (0.55, 0.65, 0.75, 0.85)]
    fs += [Filter(f"need_drawdown_{x}", f"60日回撤 <= {x}") for x in (-0.02, -0.04, -0.06, -0.08)]

    # Focused two-condition tests. Keep this deliberately small so the sweep is
    # practical and the winning rule remains explainable.
    focused = [
        "weak_cash_0.45_0.25_0.45",
        "weak_cash_0.45_0.35_0.55",
        "no_d_overheat_0.55_0.25_0.0",
        "no_d_overheat_0.55_0.35_0.0",
        "no_s_overheat_0.55_0.25_0.0",
        "no_s_overheat_0.65_0.35_0.0",
        "cool_symbol_5",
        "cool_symbol_10",
        "cool_line_symbol_5",
        "cool_line_symbol_10",
        "quality_sym_win_ge_0.46",
        "quality_sym_pf_ge_1.0",
    ]
    label_map = {f.key: f.label for f in fs}
    for i, a in enumerate(focused):
        for b in focused[i + 1:]:
            fs.append(Filter(f"{a}__{b}", f"{label_map[a]} + {label_map[b]}"))
    return fs


def rolling_select(df: pd.DataFrame, candidates: list[Filter], min_train_years: int = 3) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = sorted(df["signal_date"].dt.year.unique())
    variant_frames = {f.key: apply_filter(df, f) for f in candidates}
    selected, choices, ranks = [], [], []
    for year in years:
        train_years = [y for y in years if y < year]
        base_train = variant_frames["base"][variant_frames["base"]["signal_date"].dt.year < year]
        base_stat = summarize(base_train) if len(base_train) else {"active_rate": 1.0}
        if len(train_years) < min_train_years:
            chosen_key = "base"
            rank = pd.DataFrame()
        else:
            rows = []
            for f in candidates:
                frame = variant_frames[f.key]
                train = frame[frame["signal_date"].dt.year < year]
                st = summarize(train)
                if st["active_rate"] < max(0.45, base_stat["active_rate"] * 0.65):
                    continue
                if st["sum"] < base_stat["sum"] * 0.82:
                    continue
                if st["win"] < base_stat["win"] - 0.04:
                    continue
                st.update({"filter_key": f.key, "filter_label": f.label, "score": score(st, base_stat)})
                rows.append(st)
            rank = pd.DataFrame(rows).sort_values(["score", "sum", "worst_year_sum"], ascending=False)
            ranks.append(rank.assign(test_year=year))
            chosen_key = str(rank.iloc[0]["filter_key"]) if len(rank) else "base"
        test = variant_frames[chosen_key]
        test = test[test["signal_date"].dt.year == year].copy()
        test["test_year"] = year
        selected.append(test)
        choices.append({"test_year": year, "chosen_filter": chosen_key, "train_years": len(train_years)})
    return pd.concat(selected, ignore_index=True), pd.DataFrame(choices), pd.concat(ranks, ignore_index=True) if ranks else pd.DataFrame()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = read_base()
    candidates = filters()
    rows = []
    for f in candidates:
        frame = apply_filter(df, f)
        st = summarize(frame)
        st.update({"filter_key": f.key, "filter_label": f.label})
        rows.append(st)
    fixed_rank = pd.DataFrame(rows).sort_values(["sum", "worst_year_sum", "win"], ascending=False)
    fixed_rank.to_csv(OUT / "fixed_filter_rank.csv", index=False, encoding="utf-8-sig")

    selected, choices, ranks = rolling_select(df, candidates)
    selected.to_csv(OUT / "rolling_filter_selected_trades.csv", index=False, encoding="utf-8-sig")
    choices.to_csv(OUT / "rolling_filter_choices.csv", index=False, encoding="utf-8-sig")
    ranks.to_csv(OUT / "rolling_filter_train_rank.csv", index=False, encoding="utf-8-sig")

    base = apply_filter(df, Filter("base", "不加过滤"))
    compare_rows = []
    for name, frame in [("base_CSD_D40", base), ("rolling_filter", selected)]:
        for period, mask in [
            ("all", frame.index == frame.index),
            ("2026", frame["signal_date"].dt.year == 2026),
            ("recent_3m", frame["signal_date"] >= pd.Timestamp("2026-05-11")),
        ]:
            st = summarize(frame[mask])
            st.update({"model": name, "period": period})
            compare_rows.append(st)
    compare = pd.DataFrame(compare_rows)
    compare.to_csv(OUT / "compare.csv", index=False, encoding="utf-8-sig")

    annual = selected.groupby(selected["signal_date"].dt.year).agg(
        n=("ret2", "size"),
        active=("keep", "sum"),
        win=("ret2", lambda s: float((s[s != 0] > 0).mean()) if (s != 0).any() else np.nan),
        avg=("ret2", "mean"),
        sum=("ret2", "sum"),
    ).reset_index().rename(columns={"signal_date": "year"})
    annual.to_csv(OUT / "rolling_filter_annual.csv", index=False, encoding="utf-8-sig")

    report = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_policy": BASE_POLICY,
        "candidate_count": len(candidates),
        "formal_note": "Rolling selection uses only earlier years to select one filter for each test year.",
        "base": summarize(base),
        "rolling_filter": summarize(selected),
        "compare": compare.to_dict("records"),
        "choices": choices.to_dict("records"),
    }
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
