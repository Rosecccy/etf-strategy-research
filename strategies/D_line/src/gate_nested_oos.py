from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

import factor_dca_scan as dca
import nested_rule_oos as nested
import rolling_dca_validate as rolling
import stability_sweep as sweep


OUT = rolling.OUT
MARKET_SYMBOL = "159902"  # Long-history small/mid-cap broad-market proxy in C's clean pool.
GATE_NAMES = {
    "base": "无趋势过滤（基线）",
    "self_ma60": "自身收盘价不低于60日均线",
    "self_ma120": "自身收盘价不低于120日均线",
    "self_ma60_rising": "自身60日均线较20日前上升",
    "market_ma60": "市场代理不低于60日均线",
    "market_ret20": "市场代理近20日收益非负",
    "self_and_market_ma60": "自身与市场代理均不低于60日均线",
}


def gates(panel: pd.DataFrame) -> dict[str, np.ndarray]:
    close = pd.to_numeric(panel["close"], errors="coerce")
    self_ma60 = close.groupby(panel["symbol"], sort=False).transform(lambda item: item.rolling(60, min_periods=60).mean())
    self_ma120 = close.groupby(panel["symbol"], sort=False).transform(lambda item: item.rolling(120, min_periods=120).mean())
    self_ma60_old = self_ma60.groupby(panel["symbol"], sort=False).shift(20)

    market = panel[panel["symbol"].astype(str).str.zfill(6).eq(MARKET_SYMBOL)][["date", "close"]].copy()
    if market.empty:
        raise RuntimeError(f"Market proxy {MARKET_SYMBOL} is missing from the C ETF pool.")
    market = market.drop_duplicates("date").sort_values("date")
    market_close = pd.to_numeric(market["close"], errors="coerce")
    market_ma60 = market_close.rolling(60, min_periods=60).mean()
    market_ret20 = market_close.pct_change(20)
    dates = pd.to_datetime(panel["date"])
    market_ma60_map = dict(zip(market["date"], market_ma60))
    market_close_map = dict(zip(market["date"], market_close))
    market_ret20_map = dict(zip(market["date"], market_ret20))
    panel_market_close = dates.map(market_close_map)
    panel_market_ma60 = dates.map(market_ma60_map)
    panel_market_ret20 = dates.map(market_ret20_map)

    values = {
        "base": pd.Series(True, index=panel.index),
        "self_ma60": close >= self_ma60,
        "self_ma120": close >= self_ma120,
        "self_ma60_rising": self_ma60 >= self_ma60_old,
        "market_ma60": panel_market_close >= panel_market_ma60,
        "market_ret20": panel_market_ret20 >= 0,
    }
    values["self_and_market_ma60"] = values["self_ma60"] & values["market_ma60"]
    return {key: value.fillna(False).to_numpy(dtype=bool) for key, value in values.items()}


def run_gate(
    gate_key: str,
    gate: np.ndarray,
    full: pd.DataFrame,
    groups: dict,
    buy_conditions: dict[str, np.ndarray],
    buy_labels: dict[str, str],
    buy_cores: dict[str, str],
    sell_conditions: dict[str, np.ndarray],
    sell_labels: dict[str, str],
    sell_cores: dict[str, str],
) -> tuple[dict, list[dict]]:
    gated_buys = {identifier: mask & gate for identifier, mask in buy_conditions.items()}
    max_year = min(rolling.LAST_TEST_YEAR, int(pd.to_datetime(full["date"]).dt.year.max()))
    train_cache: dict[int, list] = {}
    for year in range(rolling.FIRST_TEST_YEAR, max_year + 1):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        train, indices = rolling.training_panel(full, cutoff)
        if train.empty or train["symbol"].nunique() < 5:
            continue
        train_cache[year] = sweep.candidate_metrics_for_year(
            train,
            gated_buys,
            sell_conditions,
            indices,
            buy_labels,
            buy_cores,
            sell_labels,
            sell_cores,
        )
    selections = {}
    for rule in sweep.RULES:
        for year, candidates in train_cache.items():
            chosen = sweep.choose(candidates, rule)
            if chosen is not None:
                selections[(rule.key, year)] = chosen

    validation_cache: dict = {}
    rows: list[dict] = []
    for year in range(nested.FIRST_NESTED_YEAR, max_year + 1):
        options = []
        for rule in sweep.RULES:
            metric = nested.validation_metric(rule, year, selections, full, groups, validation_cache)
            if metric is not None:
                options.append((rule, metric))
        if not options:
            rows.append({"gate": gate_key, "test_year": year, "status": "no_known_validation"})
            continue
        rule, validation = max(
            options,
            key=lambda item: (
                item[1]["validation_win_rate"],
                item[1]["validation_avg_return"],
                item[1]["validation_closed_batches"],
            ),
        )
        chosen = selections.get((rule.key, year))
        if chosen is None:
            rows.append({"gate": gate_key, "test_year": year, "status": "no_current_candidate", "rule": rule.key, **validation})
            continue
        buy, sell, train_metrics = chosen
        test = dca.pair_result(
            buy,
            sell,
            full,
            groups,
            entry_start=pd.Timestamp(year=year, month=1, day=1),
            entry_end=pd.Timestamp(year=year + 1, month=1, day=1),
        )
        rows.append(
            {
                "gate": gate_key,
                "test_year": year,
                "status": "selected",
                "rule": rule.key,
                **validation,
                "buy_combo": buy.label,
                "sell_combo": sell.label,
                "train_closed_batches": train_metrics["closed_batches"],
                "train_win_rate": train_metrics["win_rate"],
                "test_buy_batches": test["buy_batches"],
                "test_closed_batches": test["closed_batches"],
                "test_win_batches": test["win_batches"],
                "test_win_rate": test["win_rate"],
                "test_avg_return": test["avg_return"],
                "test_return_sum": test["return_sum"],
            }
        )
    detail = pd.DataFrame(rows)
    active = detail[detail["status"].eq("selected")].copy()
    closed = active["test_closed_batches"].fillna(0).sum()
    wins = active["test_win_batches"].fillna(0).sum()
    total = active["test_return_sum"].fillna(0).sum()
    summary = {
        "gate": gate_key,
        "gate_name": GATE_NAMES[gate_key],
        "selected_years": int(len(active)),
        "closed_batches": int(closed),
        "win_batches": int(wins),
        "oos_win_rate": float(wins / closed) if closed else np.nan,
        "oos_avg_return": float(total / closed) if closed else np.nan,
        "oos_pnl_100_cny": float(total * 100),
    }
    return summary, rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full, groups = dca.load_panel()
    buy_conditions, buy_labels, buy_cores = dca.build_conditions(full, "buy")
    sell_conditions, sell_labels, sell_cores = dca.build_conditions(full, "sell")
    all_gates = gates(full)

    summaries: list[dict] = []
    details: list[dict] = []
    for gate_key, gate in all_gates.items():
        print(f"Running gate: {gate_key}")
        summary, rows = run_gate(
            gate_key,
            gate,
            full,
            groups,
            buy_conditions,
            buy_labels,
            buy_cores,
            sell_conditions,
            sell_labels,
            sell_cores,
        )
        summaries.append(summary)
        details.extend(rows)
        print(json.dumps(summary, ensure_ascii=False))

    ranking = pd.DataFrame(summaries).sort_values(
        ["oos_win_rate", "oos_avg_return", "closed_batches"], ascending=[False, False, False]
    )
    detail = pd.DataFrame(details)
    ranking.to_csv(OUT / "gate_nested_ranking.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(OUT / "gate_nested_years.csv", index=False, encoding="utf-8-sig")
    view = ranking.copy()
    for column in ("oos_win_rate", "oos_avg_return"):
        view[column] = view[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "")
    report = [
        "# D 线：趋势过滤策略的嵌套滚动排名",
        "",
        "每一个变体都保持相同的因子、连续定投、卖出、T+1 收盘执行与年度嵌套选择器；唯一改变的是低位买入时是否要求自身或市场仍具中期趋势。",
        "",
        "市场代理为 159902 中小板 ETF，仅用于判断大盘环境，不作为交易标的或收益基准。",
        "",
        "## 排名",
        view.to_markdown(index=False),
        "",
        "注意：这是预先定义的策略变体的嵌套样本外比较。排名用于研究；正式实盘版仍需保留最后未参与策略选择的观察期。",
    ]
    (OUT / "gate_nested_report.md").write_text("\n".join(report), encoding="utf-8")
    meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market_proxy": MARKET_SYMBOL,
        "comparison": "All variants use nested annual rolling parameter selection; only the buy trend gate changes.",
    }
    (OUT / "gate_nested_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
