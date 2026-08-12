from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

import factor_dca_scan as dca
import rolling_dca_validate as rolling
import stability_sweep as sweep


OUT = rolling.OUT
FIRST_NESTED_YEAR = 2021
MIN_VALIDATION_CLOSED = 100


def validation_metric(
    rule: sweep.Rule,
    target_year: int,
    selections: dict[tuple[str, int], tuple[dca.Candidate, dca.Candidate, dict]],
    full: pd.DataFrame,
    groups: dict,
    cache: dict,
) -> dict | None:
    cutoff = pd.Timestamp(year=target_year, month=1, day=1)
    outcomes: list[dict] = []
    for test_year in range(rolling.FIRST_TEST_YEAR, target_year):
        selected = selections.get((rule.key, test_year))
        if selected is None:
            continue
        buy, sell, _ = selected
        key = (rule.key, test_year, target_year)
        if key not in cache:
            cache[key] = dca.pair_result(
                buy,
                sell,
                full,
                groups,
                entry_start=pd.Timestamp(year=test_year, month=1, day=1),
                entry_end=pd.Timestamp(year=test_year + 1, month=1, day=1),
                exit_before=cutoff,
            )
        outcomes.append(cache[key])
    closed = sum(item["closed_batches"] for item in outcomes)
    wins = sum(item["win_batches"] for item in outcomes)
    total_return = sum(item["return_sum"] for item in outcomes if np.isfinite(item["return_sum"]))
    if closed < MIN_VALIDATION_CLOSED:
        return None
    return {
        "validation_years": len(outcomes),
        "validation_closed_batches": closed,
        "validation_win_rate": wins / closed,
        "validation_avg_return": total_return / closed,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full, groups = dca.load_panel()
    buy_conditions, buy_labels, buy_cores = dca.build_conditions(full, "buy")
    sell_conditions, sell_labels, sell_cores = dca.build_conditions(full, "sell")
    max_year = min(rolling.LAST_TEST_YEAR, int(pd.to_datetime(full["date"]).dt.year.max()))

    train_cache: dict[int, list[tuple[dca.Candidate, dca.Candidate, dict]]] = {}
    for year in range(rolling.FIRST_TEST_YEAR, max_year + 1):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        train, train_indices = rolling.training_panel(full, cutoff)
        if train.empty or train["symbol"].nunique() < 5:
            continue
        train_cache[year] = sweep.candidate_metrics_for_year(
            train,
            buy_conditions,
            sell_conditions,
            train_indices,
            buy_labels,
            buy_cores,
            sell_labels,
            sell_cores,
        )
        print(f"Prepared {year}: {len(train_cache[year])} candidates")

    selections: dict[tuple[str, int], tuple[dca.Candidate, dca.Candidate, dict]] = {}
    for rule in sweep.RULES:
        for year, candidates in train_cache.items():
            chosen = sweep.choose(candidates, rule)
            if chosen is not None:
                selections[(rule.key, year)] = chosen

    validation_cache: dict = {}
    final_test_cache: dict = {}
    rows: list[dict] = []
    for target_year in range(FIRST_NESTED_YEAR, max_year + 1):
        ranked_rules: list[tuple[sweep.Rule, dict]] = []
        for rule in sweep.RULES:
            metric = validation_metric(rule, target_year, selections, full, groups, validation_cache)
            if metric is not None:
                ranked_rules.append((rule, metric))
        if not ranked_rules:
            rows.append({"test_year": target_year, "status": "no_rule_with_enough_known_validation"})
            continue
        rule, validation = max(
            ranked_rules,
            key=lambda item: (
                item[1]["validation_win_rate"],
                item[1]["validation_avg_return"],
                item[1]["validation_closed_batches"],
            ),
        )
        chosen = selections.get((rule.key, target_year))
        if chosen is None:
            rows.append({"test_year": target_year, "status": "chosen_rule_no_current_candidate", **validation, "rule": rule.key})
            continue
        buy, sell, train_metrics = chosen
        test_key = (rule.key, target_year)
        if test_key not in final_test_cache:
            final_test_cache[test_key] = dca.pair_result(
                buy,
                sell,
                full,
                groups,
                entry_start=pd.Timestamp(year=target_year, month=1, day=1),
                entry_end=pd.Timestamp(year=target_year + 1, month=1, day=1),
            )
        test = final_test_cache[test_key]
        rows.append(
            {
                "test_year": target_year,
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
        print(f"{target_year}: {rule.key}; test={test['win_rate']:.2%} ({test['closed_batches']} closed)")

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "nested_oos.csv", index=False, encoding="utf-8-sig")
    selected = result[result["status"].eq("selected")].copy()
    closed = selected["test_closed_batches"].fillna(0).sum()
    wins = selected["test_win_batches"].fillna(0).sum()
    total_return = selected["test_return_sum"].fillna(0).sum()
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "Nested annual rolling. Each year selects the selector rule using only earlier annual out-of-sample lots already closed before that year, then selects that year's buy/sell pair from pre-year data.",
        "test_years": f"{FIRST_NESTED_YEAR}-{max_year}",
        "min_known_validation_closed_batches": MIN_VALIDATION_CLOSED,
        "selected_years": int(len(selected)),
        "closed_batches": int(closed),
        "win_batches": int(wins),
        "oos_win_rate": float(wins / closed) if closed else None,
        "average_return_per_closed_batch": float(total_return / closed) if closed else None,
        "total_pnl_per_100_cny_batch": float(total_return * 100),
        "execution": "Signal is known after close; every entry and exit executes at T+1 close. Each independent lot is 100 CNY, minimum hold 7 trading days.",
        "status": "research-grade nested OOS; independent-lot result, not capital-constrained portfolio simulation",
    }
    (OUT / "nested_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    shown = selected.copy()
    for column in ("validation_win_rate", "validation_avg_return", "test_win_rate", "test_avg_return"):
        if column in shown:
            shown[column] = shown[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "")
    report = [
        "# D 线：嵌套年度滚动样本外验证",
        "",
        "每个测试年有两层历史限制：先只利用此前年度、且在当前年开始前已经卖出的实际批次，选择年度选择器约束；再仅用该测试年之前的行情，选择当年的买入与卖出组合。",
        "",
        f"- 已完成批次：{summary['closed_batches']}；盈利批次：{summary['win_batches']}；逐笔胜率：{summary['oos_win_rate']:.2%}。",
        f"- 平均单批收益：{summary['average_return_per_closed_batch']:.2%}；每批 100 元的已实现收益合计：{summary['total_pnl_per_100_cny_batch']:.2f} 元。",
        "- 这是无限资金、独立批次的统计，不能解释为单账户复利收益；未卖出的批次未计入。",
        "",
        "## 年度结果",
        shown.to_markdown(index=False) if not shown.empty else "无达到验证样本门槛的年度。",
    ]
    (OUT / "nested_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
