# C2.3 实盘输出

| 文件 | 用途 |
|---|---|
| `today_decision.csv` | 当日唯一操作建议 |
| `today_buy_signals.csv` | 全部 ETF 买入扫描结果 |
| `today_sell_checks.csv` | 当前持仓卖出检查 |
| `extrema_high_signals.csv` | 当年可用的 V1 高点信号 |
| `extrema_status.json` | 模型年份、阈值、数据日期与等价审计 |
| `extrema_exit_state.csv` | 跨日保存卖出延期状态 |
| `summary.json` | 每日决策摘要 |
| `missing_rules.csv` | 无法安全执行的缺失规则 |

高点模型只使用年初以前已经完成标签确认的数据训练。卖出延期一旦启动，会写入状态文件，程序重启后仍按原触发日继续计时。
