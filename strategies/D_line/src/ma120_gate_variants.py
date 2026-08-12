from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import factor_dca_scan as dca
from ma120_trade_audit import build_trade_log, capacity, single_account_metrics


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "ma120_gate_test"
SELECTIONS = ROOT / "out" / "ma120_selected_rules.csv"
TRAIN_YEARS = 5
MIN_TRAIN_CLOSED = 20


@dataclass(frozen=True)
class GateSpec:
    gate_id: str
    family: str
    gap_days: int
    lookback: int = 0
    rebound: float = 0.0
    up_days: int = 0


def candidate_from_ids(
    identifier: str,
    label: str,
    conditions: dict[str, np.ndarray],
    mask_override: np.ndarray | None = None,
) -> dca.Candidate:
    factors = tuple(identifier.split(" + "))
    raw_mask = np.logical_and.reduce([conditions[factor] for factor in factors])
    return dca.Candidate(identifier, label, factors, raw_mask if mask_override is None else mask_override)


def specs() -> list[GateSpec]:
    values = [
        GateSpec("baseline_ma120", "baseline", 1),
        *[
            GateSpec(f"continue_gap{gap}", "continue", gap)
            for gap in (1, 2, 3, 5, 10)
        ],
    ]
    for lookback in (3, 5, 10, 20):
        for rebound in (0.01, 0.02, 0.03, 0.05):
            for up_days in (1, 2):
                for gap in (1, 3, 5):
                    values.append(
                        GateSpec(
                            gate_id=f"rebound_l{lookback}_r{int(rebound * 100):02d}_u{up_days}_g{gap}",
                            family="rebound",
                            gap_days=gap,
                            lookback=lookback,
                            rebound=rebound,
                            up_days=up_days,
                        )
                    )
    return values


def market_context(panel: pd.DataFrame) -> dict[str, np.ndarray]:
    close = pd.to_numeric(panel["close"], errors="coerce")
    grouped = close.groupby(panel["symbol"], sort=False)
    ma120 = grouped.transform(lambda values: values.rolling(120, min_periods=120).mean())
    context: dict[str, np.ndarray] = {
        "above_ma120": (close >= ma120).fillna(False).to_numpy(dtype=bool),
        "close": close.to_numpy(dtype=float),
        "ma120": ma120.to_numpy(dtype=float),
    }
    for lookback in (3, 5, 10, 20):
        recent_low = grouped.transform(
            lambda values, size=lookback: values.rolling(size, min_periods=size).min()
        )
        context[f"off_low_{lookback}"] = (close / recent_low - 1).to_numpy(dtype=float)

    prior_1 = grouped.shift(1)
    prior_2 = grouped.shift(2)
    context["up_1"] = (close > prior_1).fillna(False).to_numpy(dtype=bool)
    context["up_2"] = ((close > prior_1) & (prior_1 > prior_2)).fillna(False).to_numpy(dtype=bool)
    return context


def cadence(mask: np.ndarray, groups: dict, gap_days: int) -> np.ndarray:
    if gap_days <= 1:
        return mask.copy()
    output = np.zeros(len(mask), dtype=bool)
    for start, end in groups["all"]:
        last_selected = -10_000
        for local in np.flatnonzero(mask[start:end]):
            if local - last_selected >= gap_days:
                output[start + local] = True
                last_selected = int(local)
    return output


def apply_gate(
    raw_buy: np.ndarray,
    spec: GateSpec,
    context: dict[str, np.ndarray],
    groups: dict,
) -> np.ndarray:
    above = context["above_ma120"]
    if spec.family == "baseline":
        return raw_buy & above

    below_raw = raw_buy & ~above
    if spec.family == "rebound":
        off_low = context[f"off_low_{spec.lookback}"]
        rising = context[f"up_{spec.up_days}"]
        below_raw &= np.isfinite(off_low) & (off_low >= spec.rebound) & rising

    below_selected = cadence(below_raw, groups, spec.gap_days)
    return (raw_buy & above) | below_selected


def build_period(
    panel: pd.DataFrame,
    groups: dict,
    buy_conditions: dict[str, np.ndarray],
    sell_conditions: dict[str, np.ndarray],
    context: dict[str, np.ndarray],
    selection: pd.Series,
    spec: GateSpec,
    entry_start: pd.Timestamp,
    entry_end: pd.Timestamp,
    test_year: int,
) -> pd.DataFrame:
    raw_buy = candidate_from_ids(
        str(selection["buy_id"]),
        str(selection["buy_combo"]),
        buy_conditions,
    )
    gated_mask = apply_gate(raw_buy.mask, spec, context, groups)
    buy = candidate_from_ids(
        str(selection["buy_id"]),
        f"{selection['buy_combo']} [{spec.gate_id}]",
        buy_conditions,
        gated_mask,
    )
    sell = candidate_from_ids(
        str(selection["sell_id"]),
        str(selection["sell_combo"]),
        sell_conditions,
    )
    trades = build_trade_log(
        buy,
        sell,
        panel,
        groups,
        entry_start,
        entry_end,
        test_year,
        str(selection["rule"]),
    )
    if not trades.empty:
        trades["gate_id"] = spec.gate_id
        trades["gate_family"] = spec.family
    return trades


def finish_log(log: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    result = log.copy()
    if result.empty:
        return result
    last_close = panel.groupby("symbol", sort=False)["close"].last().to_dict()
    result["mark_close"] = result["symbol"].map(last_close)
    result["unrealized_return"] = np.where(
        result["status"].eq("open"),
        result["mark_close"] / result["entry_close"] - 1,
        np.nan,
    )
    result["unrealized_pnl_cny"] = result["unrealized_return"] * 100
    return result


def metrics(log: pd.DataFrame, panel: pd.DataFrame) -> dict[str, float | int | str]:
    completed = log[log["status"].eq("closed")].copy()
    open_lots = log[log["status"].eq("open")].copy()
    timeline, max_total, max_symbol = capacity(log)
    realized = float(completed["pnl_cny"].sum()) if not completed.empty else 0.0
    unrealized = float(open_lots["unrealized_pnl_cny"].sum()) if not open_lots.empty else 0.0
    account = single_account_metrics(log, pd.to_datetime(panel["date"]).max()) if not log.empty else {}
    return {
        "all_buy_batches": int(len(log)),
        "completed_batches": int(len(completed)),
        "open_batches": int(len(open_lots)),
        "win_rate": float((completed["return_rate"] > 0).mean()) if not completed.empty else 0.0,
        "average_return": float(completed["return_rate"].mean()) if not completed.empty else 0.0,
        "realized_pnl_cny": realized,
        "unrealized_pnl_cny": unrealized,
        "total_mark_to_market_pnl_cny": realized + unrealized,
        "max_concurrent_batches": int(max_total),
        "max_concurrent_notional_cny": int(max_total * 100),
        "max_concurrent_single_etf": int(max_symbol),
        "account_final_value_cny": float(account.get("final_account_value_cny", 0.0)),
        "account_cagr": float(account.get("account_cagr", 0.0)),
        "minimum_initial_capital_cny": float(account.get("minimum_initial_capital_cny", 0.0)),
    }


def train_metrics(log: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, float | int]:
    if log.empty:
        return {
            "closed": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "sum_return": 0.0,
            "capital_efficiency": -np.inf,
        }
    exits = pd.to_datetime(log["exit_date"], errors="coerce")
    closed = log[log["status"].eq("closed") & exits.lt(cutoff)].copy()
    if closed.empty:
        return {
            "closed": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "sum_return": 0.0,
            "capital_efficiency": -np.inf,
        }
    _, max_total, _ = capacity(closed)
    sum_return = float(closed["return_rate"].sum())
    return {
        "closed": int(len(closed)),
        "win_rate": float((closed["return_rate"] > 0).mean()),
        "avg_return": float(closed["return_rate"].mean()),
        "sum_return": sum_return,
        "capital_efficiency": sum_return / max(max_total, 1),
    }


def annual_metrics(log: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    if log.empty:
        return pd.DataFrame()
    for year, group in log.groupby("test_year", sort=True):
        closed = group[group["status"].eq("closed")]
        open_lots = group[group["status"].eq("open")]
        realized = float(closed["pnl_cny"].sum()) if not closed.empty else 0.0
        unrealized = (
            float(open_lots["unrealized_pnl_cny"].sum()) if not open_lots.empty else 0.0
        )
        rows.append(
            {
                "model": label,
                "test_year": int(year),
                "all_buys": int(len(group)),
                "closed": int(len(closed)),
                "open": int(len(open_lots)),
                "win_rate": (
                    float((closed["return_rate"] > 0).mean()) if not closed.empty else np.nan
                ),
                "avg_return": (
                    float(closed["return_rate"].mean()) if not closed.empty else np.nan
                ),
                "realized_pnl_cny": realized,
                "unrealized_pnl_cny": unrealized,
                "mark_to_market_pnl_cny": realized + unrealized,
            }
        )
    return pd.DataFrame(rows)


def staged_metrics(
    log: pd.DataFrame,
    panel: pd.DataFrame,
    development_end: pd.Timestamp,
) -> tuple[dict[str, float | int], dict[str, float | int | str], pd.DataFrame]:
    exits = pd.to_datetime(log["exit_date"], errors="coerce")
    development = log[
        log["test_year"].le(development_end.year - 1)
        & log["status"].eq("closed")
        & exits.lt(development_end)
    ].copy()
    holdout = log[log["test_year"].ge(development_end.year)].copy()

    if development.empty:
        development_result = {
            "dev_closed": 0,
            "dev_win_rate": 0.0,
            "dev_avg_return": 0.0,
            "dev_capital_efficiency": -np.inf,
        }
    else:
        _, peak, _ = capacity(development)
        development_result = {
            "dev_closed": int(len(development)),
            "dev_win_rate": float((development["return_rate"] > 0).mean()),
            "dev_avg_return": float(development["return_rate"].mean()),
            "dev_capital_efficiency": float(development["return_rate"].sum())
            / max(peak, 1),
        }
    return development_result, metrics(holdout, panel), holdout


def fixed_test(
    panel: pd.DataFrame,
    groups: dict,
    buy_conditions: dict[str, np.ndarray],
    sell_conditions: dict[str, np.ndarray],
    context: dict[str, np.ndarray],
    selections: pd.DataFrame,
    gate_specs: list[GateSpec],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    summary_rows = []
    logs: dict[str, pd.DataFrame] = {}
    for index, spec in enumerate(gate_specs, start=1):
        parts = []
        for _, selection in selections.iterrows():
            year = int(selection["test_year"])
            part = build_period(
                panel,
                groups,
                buy_conditions,
                sell_conditions,
                context,
                selection,
                spec,
                pd.Timestamp(year=year, month=1, day=1),
                pd.Timestamp(year=year + 1, month=1, day=1),
                year,
            )
            if not part.empty:
                parts.append(part)
        log = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        log = finish_log(log, panel)
        logs[spec.gate_id] = log
        summary_rows.append({**asdict(spec), **metrics(log, panel)})
        print(f"[fixed {index}/{len(gate_specs)}] {spec.gate_id}")
    return pd.DataFrame(summary_rows), logs


def staged_select(
    fixed_logs: dict[str, pd.DataFrame],
    gate_specs: list[GateSpec],
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict], dict[str, pd.DataFrame]]:
    development_end = pd.Timestamp("2024-01-01")
    rows = []
    holdout_logs = {}
    spec_lookup = {spec.gate_id: spec for spec in gate_specs}
    for gate_id, log in fixed_logs.items():
        development, holdout, holdout_log = staged_metrics(
            log,
            panel,
            development_end,
        )
        holdout_logs[gate_id] = holdout_log
        rows.append(
            {
                **asdict(spec_lookup[gate_id]),
                **development,
                **{f"holdout_{key}": value for key, value in holdout.items()},
            }
        )
    table = pd.DataFrame(rows)
    eligible = table[table["dev_closed"].ge(MIN_TRAIN_CLOSED)].copy()
    win_choice = eligible.sort_values(
        ["dev_win_rate", "dev_avg_return", "dev_closed"],
        ascending=False,
    ).iloc[0]
    efficiency_choice = eligible.sort_values(
        ["dev_capital_efficiency", "dev_win_rate", "dev_avg_return"],
        ascending=False,
    ).iloc[0]
    choices = {
        "win": win_choice.to_dict(),
        "efficiency": efficiency_choice.to_dict(),
    }
    chosen_logs = {
        mode: holdout_logs[str(choice["gate_id"])]
        for mode, choice in choices.items()
    }
    return table, choices, chosen_logs


def rolling_select(
    mode: str,
    panel: pd.DataFrame,
    groups: dict,
    buy_conditions: dict[str, np.ndarray],
    sell_conditions: dict[str, np.ndarray],
    context: dict[str, np.ndarray],
    selections: pd.DataFrame,
    gate_specs: list[GateSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    choices = []
    test_parts = []
    for _, selection in selections.iterrows():
        year = int(selection["test_year"])
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        train_start = pd.Timestamp(year=year - TRAIN_YEARS, month=1, day=1)
        candidates = []
        for spec in gate_specs:
            train_log = build_period(
                panel,
                groups,
                buy_conditions,
                sell_conditions,
                context,
                selection,
                spec,
                train_start,
                cutoff,
                year,
            )
            stat = train_metrics(train_log, cutoff)
            candidates.append((spec, stat))

        eligible = [
            item for item in candidates if int(item[1]["closed"]) >= MIN_TRAIN_CLOSED
        ]
        if not eligible:
            eligible = [
                item for item in candidates if item[0].gate_id == "baseline_ma120"
            ]
        if mode == "win":
            chosen, chosen_stat = max(
                eligible,
                key=lambda item: (
                    item[1]["win_rate"],
                    item[1]["avg_return"],
                    item[1]["closed"],
                ),
            )
        else:
            chosen, chosen_stat = max(
                eligible,
                key=lambda item: (
                    item[1]["capital_efficiency"],
                    item[1]["win_rate"],
                    item[1]["avg_return"],
                ),
            )

        test_log = build_period(
            panel,
            groups,
            buy_conditions,
            sell_conditions,
            context,
            selection,
            chosen,
            cutoff,
            pd.Timestamp(year=year + 1, month=1, day=1),
            year,
        )
        if not test_log.empty:
            test_log["rolling_mode"] = mode
            test_parts.append(test_log)
        choices.append(
            {
                "mode": mode,
                "test_year": year,
                **asdict(chosen),
                **{f"train_{key}": value for key, value in chosen_stat.items()},
            }
        )
        print(f"[rolling {mode}] {year}: {chosen.gate_id}")

    test = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()
    return pd.DataFrame(choices), finish_log(test, panel)


def case_study(
    panel: pd.DataFrame,
    groups: dict,
    buy_conditions: dict[str, np.ndarray],
    context: dict[str, np.ndarray],
    selections: pd.DataFrame,
    rolling_choices: pd.DataFrame,
) -> pd.DataFrame:
    selection = selections.loc[selections["test_year"].eq(2025)].iloc[0]
    raw_buy = candidate_from_ids(
        str(selection["buy_id"]),
        str(selection["buy_combo"]),
        buy_conditions,
    ).mask
    symbol_mask = panel["symbol"].astype(str).str.zfill(6).eq("159811").to_numpy()
    dates = pd.to_datetime(panel["date"])
    date_mask = dates.between("2025-03-24", "2025-04-15").to_numpy()
    close = context["close"]
    ma120 = context["ma120"]
    frame = pd.DataFrame(
        {
            "date": dates,
            "symbol": panel["symbol"].astype(str).str.zfill(6),
            "close": close,
            "ma120": ma120,
            "raw_low_combo": raw_buy,
            "baseline_signal": apply_gate(
                raw_buy,
                GateSpec("baseline_ma120", "baseline", 1),
                context,
                groups,
            ),
        }
    )
    for mode in ("win", "efficiency"):
        row = rolling_choices[
            rolling_choices["mode"].eq(mode)
            & rolling_choices["test_year"].eq(2025)
        ].iloc[0]
        chosen = GateSpec(
            str(row["gate_id"]),
            str(row["family"]),
            int(row["gap_days"]),
            int(row["lookback"]),
            float(row["rebound"]),
            int(row["up_days"]),
        )
        frame[f"{mode}_signal"] = apply_gate(raw_buy, chosen, context, groups)
        frame[f"{mode}_gate"] = chosen.gate_id
    return frame.loc[symbol_mask & date_mask].copy()


def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)

    panel, groups = dca.load_panel()
    buy_conditions, _, _ = dca.build_conditions(panel, "buy")
    sell_conditions, _, _ = dca.build_conditions(panel, "sell")
    context = market_context(panel)
    selections = pd.read_csv(SELECTIONS, encoding="utf-8-sig")
    gate_specs = specs()

    fixed, fixed_logs = fixed_test(
        panel,
        groups,
        buy_conditions,
        sell_conditions,
        context,
        selections,
        gate_specs,
    )
    fixed.to_csv(OUT / "fixed_variants.csv", index=False, encoding="utf-8-sig")

    staged_table, staged_choices, staged_logs = staged_select(
        fixed_logs,
        gate_specs,
        panel,
    )
    staged_table.to_csv(
        OUT / "staged_variants.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUT / "staged_choices.json").write_text(
        json.dumps(staged_choices, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    for mode, log in staged_logs.items():
        log.to_csv(
            OUT / f"staged_{mode}_holdout_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )

    all_choices = []
    rolling_logs = {}
    for mode in ("win", "efficiency"):
        choices, log = rolling_select(
            mode,
            panel,
            groups,
            buy_conditions,
            sell_conditions,
            context,
            selections,
            gate_specs,
        )
        all_choices.append(choices)
        rolling_logs[mode] = log
        log.to_csv(OUT / f"rolling_{mode}_trades.csv", index=False, encoding="utf-8-sig")

    choices = pd.concat(all_choices, ignore_index=True)
    choices.to_csv(OUT / "rolling_choices.csv", index=False, encoding="utf-8-sig")

    rolling_summary = []
    for mode, log in rolling_logs.items():
        rolling_summary.append({"mode": mode, **metrics(log, panel)})
    rolling_summary_frame = pd.DataFrame(rolling_summary)
    rolling_summary_frame.to_csv(
        OUT / "rolling_summary.csv", index=False, encoding="utf-8-sig"
    )

    annual = [
        annual_metrics(fixed_logs["baseline_ma120"], "baseline_ma120"),
        annual_metrics(rolling_logs["win"], "rolling_win"),
        annual_metrics(rolling_logs["efficiency"], "rolling_efficiency"),
    ]
    for mode, log in staged_logs.items():
        annual.append(
            annual_metrics(
                log,
                f"staged_{mode}_{staged_choices[mode]['gate_id']}",
            )
        )
    pd.concat(annual, ignore_index=True).to_csv(
        OUT / "annual_compare.csv",
        index=False,
        encoding="utf-8-sig",
    )

    case = case_study(
        panel,
        groups,
        buy_conditions,
        context,
        selections,
        choices,
    )
    case.to_csv(OUT / "case_159811_2025.csv", index=False, encoding="utf-8-sig")

    best_fixed_profit = fixed.sort_values(
        ["total_mark_to_market_pnl_cny", "win_rate"],
        ascending=False,
    ).iloc[0]
    best_fixed_win = fixed.loc[fixed["completed_batches"].ge(20)].sort_values(
        ["win_rate", "average_return", "completed_batches"],
        ascending=False,
    ).iloc[0]
    baseline = fixed.loc[fixed["gate_id"].eq("baseline_ma120")].iloc[0]

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "control_variables": (
            "Original annual rolling buy/sell combos, T+1 execution, fixed 100-CNY "
            "lots and exits are unchanged. Only the MA120 entry gate changes."
        ),
        "candidate_count": len(gate_specs),
        "training_window_years": TRAIN_YEARS,
        "minimum_training_closed_batches": MIN_TRAIN_CLOSED,
        "baseline": baseline.to_dict(),
        "best_fixed_profit_research_only": best_fixed_profit.to_dict(),
        "best_fixed_win_research_only": best_fixed_win.to_dict(),
        "staged_holdout": staged_choices,
        "rolling": {
            row["mode"]: row
            for row in rolling_summary_frame.to_dict(orient="records")
        },
        "warning": (
            "Fixed best variants use the whole test interval for ranking and are research-only. "
            "The annual rolling and staged holdout results are causal comparisons."
        ),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    for gate_id in {
        "baseline_ma120",
        str(best_fixed_profit["gate_id"]),
        str(best_fixed_win["gate_id"]),
    }:
        fixed_logs[gate_id].to_csv(
            OUT / f"fixed_{gate_id}_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )

    report = [
        "# MA120 买入门槛控制变量测试",
        "",
        "保持原年度滚动买卖组合、卖点、T+1执行和每批100元不变，只改变买入门槛。",
        "",
        "## 严格滚动结果",
        "",
        rolling_summary_frame.to_markdown(index=False),
        "",
        "## 独立留出期结果",
        "",
        "使用 2021-2023 年选择固定门槛，2024-2026 年仅用于留出检验。",
        "",
        pd.DataFrame(staged_choices.values()).to_markdown(index=False),
        "",
        "## 每年选择",
        "",
        choices.to_markdown(index=False),
        "",
        "## 说明",
        "",
        "- fixed_variants.csv 的全区间最优仅用于发现规律，不能直接晋升。",
        "- staged_* 使用 2021-2023 年选参，2024-2026 年作为独立留出期。",
        "- rolling_win 每年按过去5年胜率优先选择。",
        "- rolling_efficiency 每年按过去5年的单位峰值资金收益优先选择。",
        "- 结果未计手续费；资金口径与原 MA120 审计保持一致。",
    ]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
