# ETF Strategy Research

本仓库的当前正式基线为 **2026-08-21 audited 14:45 release V2**。当前正式策略线是 **C / S / D / R**；`strategies/` 下的 C/S/D/V 目录仅作为历史兼容底座，不代表当前最终基线。

## 当前正式结果

| 策略线 | 最终资金 | 交易数 | 胜率 | 平均单笔收益 | 最大回撤 | 2024+ 审计胜率 |
|---|---:|---:|---:|---:|---:|---:|
| C | 209,530.06 | 38 | 78.95% | 8.93% | -13.68% | 88.89% |
| S | 149,989.34 | 44 | 77.27% | 6.81% | -12.70% | 81.25% |
| D | 656,342.50 | 1303 | 66.46% | 4.96% | -18.57% | 67.86% |
| R | 232,901.11 | 44 | 77.27% | 7.96% | -13.68% | 78.57% |

数值来自 `same_day_1445/release_v2/selected_summary.csv`。2024+ 已被多次查看，因此这里只称为后段审计窗口，不称为全新未见测试。

## 当前可直接验证的内容

仓库继续保持轻量结构：C/S/R 的 126 笔冻结交易保存在 `selected_trades_csr.csv`；D 的 1303 笔交易以逐年聚合 `d_annual_compact.csv` 加 6 笔跨年边界证据 `d_cross_year_entry_audit.csv` 复核。因此 GitHub 内可以验证：

- C/S/R 的逐笔交易数、胜率、平均收益、期末资金、最大回撤和 2024+ 审计摘要；
- D 的总交易数、年度交易数、胜率、平均收益、research-sum 期末值和 2024+ 聚合摘要；
- C/S/R 的单仓不重叠约束；
- D 线按有效入场年度统计的年度交易数；
- `selected_choices.csv` 中每个测试年度的训练年份是否严格早于测试年度，并与配置窗口完全一致；
- `audit.json`、`baseline.json`、年度汇总与交易表是否一致。

运行：

```bash
python scripts/verify_current_baseline.py
python scripts/verify_package.py
python -m pytest -q tests/test_v2_audit_consistency.py
```

## D 线年度口径

V2 的年度统计统一使用实际执行后的 `entry` 年份：

- `year` / `entry_year`：有效入场年份，用于年度统计、2024+ 审计和滚动选择结果核对；
- `source_year`：原始正式账本中 `old_entry` 的年份，只用于追溯原信号。

这样可以避免跨年移动交易把 12 月 31 日的实际入场误记到下一年。

## 因果选择口径

`selected_choices.csv` 现在显式记录 `train_years`、`train_year_count`、`train_start_year`、`train_end_year` 和 `causal_train_window`。验证脚本会按每条线的 `window` 与 `min_years` 独立重算预期训练年份，并要求所有训练年份严格满足 `< test_year`。

这能验证已冻结选择证据的因果边界。若要从原始行情重新搜索全部候选政策，仍需恢复 2026-08-21 完整审计包中的原始行情、Parquet 特征面板及深度研究脚本。

## 仓库结构

- `same_day_1445/release_v2/`：当前冻结 V2 基线、C/S/R 逐笔账本、D 年度与跨年边界证据；
- `scripts/verify_current_baseline.py`：当前 V2 的主验证入口；
- `scripts/baseline_audit.py`：审计规则的纯函数实现；
- `docs/`：复核、重建、优化和仓库边界说明；
- `strategies/`：历史 C/S/D/V 兼容代码和数据底座，仅用于旧版本复核；
- `run_all_strategies.py`、`scripts/make_repo_runnable.py`、`scripts/verify_runnable.py`：历史兼容工具，不用于证明当前 C/S/D/R V2 基线。

## 完整重建边界

GitHub 不重复提交 D 的完整 1303 笔研究账本，也不提交数百 MB 的原始行情、候选面板、Parquet 特征和模型二进制。D 的逐笔最大回撤重新计算，以及完整搜索与深度审计，仍需从原始 2026-08-21 审计包恢复这些工件。不要用后来下载的数据静默替换历史输入。详见 `docs/DATA_RECONSTRUCTION.md`。

## 后续优化规则

后续实验必须以 `same_day_1445/release_v2` 为冻结基线，并遵守 `docs/OPTIMIZATION_PROTOCOL.md`：测试年度只能由更早年度选择参数；同时报告收益、胜率、回撤和频率；不得使用未来高低点作为当日输入；2024+ 不得重新包装为 pristine holdout。

本仓库仅用于研究，不构成投资建议。
