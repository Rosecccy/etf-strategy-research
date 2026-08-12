# 四策略私有复核仓库

这个包用于把当前四条 ETF 策略线交给同学复核或继续研究。它不是投资建议，只是研究工程包。

## 四个策略文件夹

- `strategies/C_line`：C 线，30 只干净 ETF 池，稀有强反弹/稳健升级线。
- `strategies/S_line`：S 线，扩展干净 ETF 池，趋势补充线。
- `strategies/D_line`：D 线，当前主力 C/S/D 组合与严格滚动卖出保护。
- `strategies/V_line`：V 线，恐惧/贪婪、强制 Top1 与频率保持相关研究线。

## 快速自检

```powershell
python scripts/verify_package.py
```

如果显示 `PACKAGE OK`，说明关键数据、代码、结果和说明都在。

## 复核顺序

1. 先读 `docs/REPRODUCE.md`。
2. 再读每条线自己的 `README.md`。
3. 用 `file_manifest_sha256.csv` 核对文件完整性。
4. 复核结果时优先看各线 `results/` 下的最终交易表和摘要。

## 关于被排除的大文件

GitHub 普通仓库单文件上限约 100MB。本包故意不放超大中间候选缓存表，例如完整候选面板。正式结果复核所需的原始干净数据、配置、代码和最终交易明细已保留；若要重新跑全量搜索，可用 `src/` 脚本从原始 ETF 数据再生成中间表。


## 离线运行检查

这个仓库现在同时保留两套结构：

- `strategies/C_line`、`strategies/S_line`、`strategies/D_line`、`strategies/V_line`：给人阅读和复核的精简结构。
- `C`、`S`、`D`、`V`：给旧脚本直接运行的兼容结构，路径名与原工程一致。

同学拿到仓库后，先在仓库根目录运行：

```bash
python scripts/verify_package.py
python scripts/verify_runnable.py
python run_all_strategies.py
```

其中 `verify_runnable.py` 会在不依赖旧电脑目录的情况下读取本仓库内的数据、配置和结果，确认能独立跑出四条策略的当前摘要。
