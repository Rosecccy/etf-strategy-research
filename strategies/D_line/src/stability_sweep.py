from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

import factor_dca_scan as dca
import rolling_dca_validate as rolling


OUT = rolling.OUT


@dataclass(frozen=True)
class Rule:
    min_closed: int
    min_closed_rate: float
    min_covered_years: int
    min_worst_year_win: float

    @property
    def key(self) -> str:
        return (
            f"n{self.min_closed}_exit{int(self.min_closed_rate * 100)}_"
            f"years{self.min_covered_years}_worst{int(self.min_worst_year_win * 100)}"
        )


RULES = [
    Rule(*values)
    for values in itertools.product(
        (40, 80, 120),
        (0.60, 0.75, 0.85),
        (0, 2, 3),
        (0.00, 0.45, 0.55),
    )
]


def candidate_metrics_for_year(
    train: pd.DataFrame,
    full_buy_conditions: dict[str, np.ndarray],
    full_sell_conditions: dict[str, np.ndarray],
    train_indices: np.ndarray,
    buy_labels: dict[str, str],
    buy_cores: dict[str, str],
    sell_labels: dict[str, str],
    sell_cores: dict[str, str],
) -> list[tuple[dca.Candidate, dca.Candidate, dict]]:
    train_buy_conditions = {key: value[train_indices] for key, value in full_buy_conditions.items()}
    train_sell_conditions = {key: value[train_indices] for key, value in full_sell_conditions.items()}
    buys, _, _ = dca.candidates_for_side(
        train_buy_conditions, buy_labels, buy_cores, train, "buy", max_factor_count=3
    )
    sells, _, _ = dca.candidates_for_side(
        train_sell_conditions, sell_labels, sell_cores, train, "sell", max_factor_count=2
    )
    groups = rolling.group_slices(train)
    results: list[tuple[dca.Candidate, dca.Candidate, dict]] = []
    for train_buy, train_sell in itertools.product(
        buys[: rolling.ROLLING_BUY_LIMIT], sells[: rolling.ROLLING_SELL_LIMIT]
    ):
        metrics = dca.pair_result(train_buy, train_sell, train, groups, with_years=True)
        if metrics["closed_batches"] >= 40 and metrics["closed_rate"] >= 0.60:
            results.append(
                (
                    rolling.expand_candidate(train_buy, full_buy_conditions),
                    rolling.expand_candidate(train_sell, full_sell_conditions),
                    metrics,
                )
            )
    return results


def choose(results: list[tuple[dca.Candidate, dca.Candidate, dict]], rule: Rule):
    eligible = [
        item
        for item in results
        if item[2]["closed_batches"] >= rule.min_closed
        and item[2]["closed_rate"] >= rule.min_closed_rate
        and item[2].get("covered_years", 0) >= rule.min_covered_years
        and np.isfinite(item[2].get("worst_year_win_rate", np.nan))
        and item[2]["worst_year_win_rate"] >= rule.min_worst_year_win
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (
            item[2]["win_rate"],
            item[2]["worst_year_win_rate"],
            item[2]["avg_return"],
            item[2]["closed_batches"],
        ),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full, groups = dca.load_panel()
    buy_conditions, buy_labels, buy_cores = dca.build_conditions(full, "buy")
    sell_conditions, sell_labels, sell_cores = dca.build_conditions(full, "sell")
    max_year = min(rolling.LAST_TEST_YEAR, int(pd.to_datetime(full["date"]).dt.year.max()))
    train_cache: dict[int, list[tuple[dca.Candidate, dca.Candidate, dict]]] = {}
    test_cache: dict[tuple[int, str, str], dict] = {}

    for year in range(rolling.FIRST_TEST_YEAR, max_year + 1):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        train, train_indices = rolling.training_panel(full, cutoff)
        if train.empty or train["symbol"].nunique() < 5:
            continue
        train_cache[year] = candidate_metrics_for_year(
            train,
            buy_conditions,
            sell_conditions,
            train_indices,
            buy_labels,
            buy_cores,
            sell_labels,
            sell_cores,
        )
        print(f"Prepared {year}: {len(train_cache[year])} train candidates")

    summary_rows: list[dict] = []
    year_rows: list[dict] = []
    for rule in RULES:
        selections = []
        for year, train_results in train_cache.items():
            chosen = choose(train_results, rule)
            if chosen is None:
                continue
            buy, sell, train_metrics = chosen
            cache_key = (year, buy.identifier, sell.identifier)
            if cache_key not in test_cache:
                test_cache[cache_key] = dca.pair_result(
                    buy,
                    sell,
                    full,
                    groups,
                    entry_start=pd.Timestamp(year=year, month=1, day=1),
                    entry_end=pd.Timestamp(year=year + 1, month=1, day=1),
                )
            test = test_cache[cache_key]
            selections.append(test)
            year_rows.append(
                {
                    "rule": rule.key,
                    "test_year": year,
                    "buy_combo": buy.label,
                    "sell_combo": sell.label,
                    "train_win_rate": train_metrics["win_rate"],
                    "train_worst_year_win_rate": train_metrics.get("worst_year_win_rate"),
                    "train_avg_return": train_metrics["avg_return"],
                    "train_closed_batches": train_metrics["closed_batches"],
                    "test_closed_batches": test["closed_batches"],
                    "test_win_batches": test["win_batches"],
                    "test_win_rate": test["win_rate"],
                    "test_avg_return": test["avg_return"],
                    "test_return_sum": test["return_sum"],
                }
            )
        closed = sum(item["closed_batches"] for item in selections)
        wins = sum(item["win_batches"] for item in selections)
        total_return = sum(item["return_sum"] for item in selections if np.isfinite(item["return_sum"]))
        summary_rows.append(
            {
                "rule": rule.key,
                "selected_years": len(selections),
                "closed_batches": closed,
                "win_batches": wins,
                "oos_win_rate": wins / closed if closed else np.nan,
                "oos_avg_return": total_return / closed if closed else np.nan,
                "oos_pnl_100_cny": total_return * 100,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["oos_win_rate", "oos_avg_return", "closed_batches"], ascending=[False, False, False]
    )
    details = pd.DataFrame(year_rows)
    summary.to_csv(OUT / "stability_sweep.csv", index=False, encoding="utf-8-sig")
    details.to_csv(OUT / "stability_sweep_years.csv", index=False, encoding="utf-8-sig")
    top = summary.head(20).copy()
    for column in ("oos_win_rate", "oos_avg_return"):
        top[column] = top[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "")
    report = [
        "# D 线：年度选择器稳定性邻域扫描",
        "",
        "扫描的仅是年度训练期筛选约束：训练已完成批次数、退出完成率、覆盖年份数、最差年度胜率。买入/卖出定义、T+1 收盘执行、单批 100 元和最短持有 7 日均未改变。",
        "",
        "**注意**：下表可用于发现稳定参数区域，但不能把其中的样本外最高行直接称为正式规则，因为它仍然使用整段样本外结果来比较参数。正式规则还需要下一层嵌套滚动验证。",
        "",
        "## 按样本外逐笔胜率排序的前 20 个筛选约束",
        top.to_markdown(index=False),
    ]
    (OUT / "stability_report.md").write_text("\n".join(report), encoding="utf-8")
    meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rules_tested": len(RULES),
        "years_prepared": sorted(train_cache),
        "formal_status": "research only; parameter choice needs nested rolling validation before deployment",
    }
    (OUT / "stability_sweep_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
