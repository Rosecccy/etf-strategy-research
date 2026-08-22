# V2 复核与重建说明

当前正式基线是 `same_day_1445/release_v2`，策略线为 C/S/D/R。`strategies/` 下的 C/S/D/V 属于历史兼容层，不再作为当前结果来源。

## 1. GitHub 内的冻结结果复核

在仓库根目录执行：

```bash
python scripts/verify_current_baseline.py
python scripts/verify_package.py
python -m pytest -q tests/test_v2_audit_consistency.py
```

主验证会逐笔重算 C/S/R 的 126 笔冻结交易，并将 D 的 `d_annual_compact.csv` 与逐年度 `selected_choices.csv` 交叉核对；6 笔跨年移动交易由 `d_cross_year_entry_audit.csv` 单独验证。它同时核对年度训练窗口、`selected_summary.csv`、`audit.json` 和 `baseline.json`。D 的逐笔最大回撤重新计算仍属于完整源包复核范围。

## 2. 因果选择证据

`selected_choices.csv` 的训练窗口字段必须满足：

- 所有 `train_years < test_year`；
- 实际训练年份与 `window`、`min_years` 推导出的预期年份完全一致；
- 历史年度不足时必须使用该线的 protected base policy。

这验证的是冻结选择证据。若需要重新搜索所有候选政策，则进入第 3 步。

## 3. 全量重新生成

完整重新生成需要从 2026-08-21 原始审计包恢复原始行情、Parquet 特征面板、正式 C/S/D/R 账本和深度研究脚本，然后从原始数据重跑 14:45 搜索、发布与审计。

GitHub 中没有这些大型输入，因此不会提供一个缺少输入却声称能完整重建的命令。恢复规则见 `DATA_RECONSTRUCTION.md`。

## 4. 历史兼容层

只有在复核旧 C/S/D/V 工程时才使用：

```bash
python scripts/make_repo_runnable.py
python scripts/verify_runnable.py
python run_all_strategies.py
```

这些命令不验证当前 C/S/D/R V2 正式结果。
