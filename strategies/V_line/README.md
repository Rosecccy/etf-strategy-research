# V线：恐惧贪婪与强制排名研究线

## 目录

- `data/`：该策略所需的干净 ETF 数据、池子和质量检查信息。
- `config/`：策略配置。
- `src/`：可继续研究和复核的核心脚本。
- `results/`：正式结果、交易表、年度表或摘要。
- `live/`：若存在，则为最近一次实盘/每日决策输出。

## 数据说明

V 线当前复用 C 线干净 ETF 数据池，重点保存恐惧/贪婪、强制 Top1、频率保持相关结果。

## 结果说明

主要结果在 `results/fear_greed/`、`results/forced_top1_confidence_gate/` 与 `results/frequency_preserving_upgrade/`。

## 复核建议

先看 `results/` 中的摘要文件，再查看交易明细。若要继续调参，优先复制本目录为新实验目录，不要直接覆盖正式结果。
