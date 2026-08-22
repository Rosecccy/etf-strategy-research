# Data Reconstruction

当前 GitHub 仓库采用“完整冻结小账本 + 可恢复大型研究工件”的结构。

## GitHub 中直接保留

- `same_day_1445/release_v2/selected_trades_csr.csv`：C/S/R 共 126 笔冻结交易；
- `d_annual_compact.csv` 与 `d_cross_year_entry_audit.csv`：D 线年度汇总与跨年边界证据；
- `selected_choices.csv`：逐年度选择及训练年份证据；
- `selected_summary.csv`、`d_annual_compact.csv`、`audit.json`、`baseline.json`；
- 当前验证脚本、回归测试和优化协议；
- 历史 `strategies/` 兼容底座。

## GitHub 中不重复提交

- 数百 MB 原始或重复行情归档；
- Parquet 特征面板；
- 模型二进制；
- 大型候选搜索面板与可再生中间缓存；
- D 线完整 1303 笔逐笔研究账本。

## 全量复跑时恢复

需要从 2026-08-21 原始审计压缩包恢复对应原始行情、C/S/D/R 正式源账本、特征面板和深度研究脚本，再执行原始搜索与发布流程。禁止用后来下载的数据静默替换历史输入后仍沿用旧版本号。

## 年度口径

D 线跨年移动交易以实际 `entry` 年份作为 `year` / `entry_year`，原始账本年份保留在 `source_year`。年度汇总、2024+ 审计窗口和当前 GitHub 验证均采用有效入场年份。
