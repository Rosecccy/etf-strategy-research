# ETF Strategy Research

本仓库的当前正式基线已更新为 **2026-08-21 audited 14:45 release V2**。

为避免把数百 MB 的重复行情、Parquet 特征面板和模型二进制继续堆进 Git，仓库采用两层结构：

- `same_day_1445/release_v2/`：**当前正式基线**，后续所有优化都从这里比较。
- `strategies/`：历史可运行策略底座和数据，保留用于兼容、复核与重建，不再代表当前最终表现。

## 当前正式结果

| 策略线 | 最终资金 | 交易数 | 胜率 | 平均单笔收益 | 最大回撤 | 2024+ Holdout 胜率 |
|---|---:|---:|---:|---:|---:|---:|
| C | 209,530.06 | 38 | 78.95% | 8.93% | -13.68% | 88.89% |
| S | 149,989.34 | 44 | 77.27% | 6.81% | -12.70% | 81.25% |
| D | 656,342.50 | 1303 | 66.46% | 4.96% | -18.57% | 67.86% |
| R | 232,901.11 | 44 | 77.27% | 7.96% | -13.68% | 78.57% |

上述结果来自 `same_day_1445/release_v2/selected_summary.csv`。

## 先运行什么

```bash
python -m pip install -r requirements.txt
python scripts/verify_current_baseline.py
```

若要复核完整 14:45 发布逻辑：

```bash
python scripts/audit_1445_release_v2.py
python scripts/audit_three_ledgers.py
```

GitHub 轻量版不重复提交完整 1429 笔 V2 交易大表：C/S/R 的 126 笔交易保留在 `selected_trades_csr.csv`，D 线保留逐年聚合；完整账本在原始审计包中，并可由发布脚本重建。部分深度复跑需要完整归档中的原始行情或派生特征文件。省略数据的原则和恢复方式见 `docs/DATA_RECONSTRUCTION.md`。

## 后续优化规则

后续实验必须以 `same_day_1445/release_v2` 为冻结基线，遵守 `docs/OPTIMIZATION_PROTOCOL.md`：

1. 测试年份只能由更早年份选择参数、阈值、模型和路由；
2. 同时报告胜率、收益、最大回撤、交易频率、持仓/空仓时间；
3. 不允许用未来高低点作为当日输入；
4. 2024+ Holdout 已被多次查看，不再称为“全新未见测试”；
5. 159663 的已知价格口径断层继续保留，并在相关结果中单独标记。

本仓库仅用于研究，不构成投资建议。
