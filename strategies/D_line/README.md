# D线：主力 C/S/D 组合与卖出保护

## 目录

- `data/`：该策略所需的干净 ETF 数据、池子和质量检查信息。
- `config/`：策略配置。
- `src/`：可继续研究和复核的核心脚本。
- `results/`：正式结果、交易表、年度表或摘要。
- `live/`：若存在，则为最近一次实盘/每日决策输出。

## 数据说明

D 线复用 C 线的 30 只干净 ETF 数据池，并保留严格滚动正式选择结果。

## 结果说明

正式策略文件在 `results/formal/current_strategy.json`；正式交易明细在 `results/formal/formal_selected_trades.csv`。

## 复核建议

先看 `results/` 中的摘要文件，再查看交易明细。若要继续调参，优先复制本目录为新实验目录，不要直接覆盖正式结果。
