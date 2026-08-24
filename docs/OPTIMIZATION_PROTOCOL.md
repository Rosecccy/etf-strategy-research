# Optimization Protocol

## Frozen baseline

基线版本：`same_day_1445/release_v2`，冻结日期 2026-08-21。

| Line | Win rate | Avg trade | Max drawdown | Trades |
|---|---:|---:|---:|---:|
| C | 78.95% | 8.93% | -13.68% | 38 |
| S | 77.27% | 6.81% | -12.70% | 44 |
| D | 66.46% | 4.96% | -18.57% | 1303 |
| R | 77.27% | 7.96% | -13.68% | 44 |

## Causal rolling rule

对测试年度 Y 的任何参数、阈值、权重、模型结构和路由选择，只能使用 `< Y` 的样本。真实高低点可以用于标签和事后评分，但不能进入 Y 当日可用特征。

当前 V2 的 `selected_choices.csv` 必须保存明确的训练年份证据，且 `scripts/verify_current_baseline.py` 必须能够独立重算预期窗口并确认所有训练年份早于测试年份。

对 live self-optimizer，新交易日 T 的 14:45 决策只能使用截至 T 14:45 已经可观察的数据；收盘价、之后分钟数据和最终交易收益只能在其真正可观察之后用于对账、标签和后续优化，不能回填到此前决策。

## Accounting conventions

- C/S/R：单仓复利；
- D：保留原研究账本的 research-sum 口径，不与 C/S/R 的期末资金直接横比；
- 年度统计统一按实际 `entry` 年份；`source_year` 只用于追溯原始正式账本；
- 2024+ 是已查看过的后段审计窗口，不是 pristine holdout；
- live 候选必须与 formal 使用同一个 `opportunity_id` 做可比评估；候选过滤掉的机会仍以 `triggered=0` 保留在触发率分母中。

## Historical promotion guardrails

历史研究候选必须逐年和总体同时报告：胜率、平均单笔净收益、期末/累计收益、最大回撤、交易次数与年均频率、平均持有天数、平均空仓率/资金占用。

历史研究默认约束：

- 最大回撤不得比对应基线恶化超过 2 个百分点；
- 交易数不得低于基线的 80%，除非胜率和收益的提升足以明确解释机会过滤；
- 胜率不得下降超过 1 个百分点，除非收益与回撤同时改善；
- 至少 2/3 的后段滚动窗口取得正向综合改善；
- 参数不得因为看过目标测试年的结果而回填；
- 候选版本必须通过当前基线一致性检查和新增回归测试后才能进入 PR。

这些历史约束不能替代 live self-optimizer 的更严格自动晋级门槛。

## Live self-optimizer forward gate

`opt/live-self-optimizer-v1` 实现程序化前向采样、shadow 比较、自动门控、晋级和回滚。默认部署状态为 `SHADOW_ONLY`；`same_day_1445/release_v2` 永远保留为冻结锚点。

默认自动晋级门槛为：

- 触发保留率 `>= 90%`；
- 滚动胜率相对当前 formal 至少 `+0.5` 个百分点；
- 平均单笔收益不得下降；
- 最大回撤相对 formal 恶化不得超过 `1` 个百分点；
- 必须达到配置要求的非空 forward evaluation windows；
- 最近评估窗口至少 `2/3` 综合改善，最近窗口不得明显退化；
- 参数邻域必须形成稳定平台，不接受孤立最优点；
- D 至少新增 100 笔已闭合、此前未用于上一次晋级的 forward evidence；C/S/R 至少新增 15 笔并覆盖至少两个程序化市场状态；
- 晋级与再次晋级之间必须满足 cooldown；
- 候选不仅需要胜过当前 formal，在 formal 已经不是 V2 时仍必须在同一机会集合上继续通过 V2 frozen-anchor guard；
- formal 与候选当前持仓状态不兼容时只允许进入 `PENDING_TRANSITION`，不得持仓中途强制切换；
- 严重漂移、数据质量失败、因果审计失败或未完成收盘对账均禁止晋级并可强制进入 `SHADOW_ONLY`/`DATA_HOLD`。

每次正式晋级都会生成不可变 release manifest，记录父版本、candidate、时间和 gate evidence。上一次晋级之前的样本不会再次计入下一次 `minimum_new_samples`。

## Automatic rollback

自动晋级后的 formal 会继续与其父版本在晋级后新增的同机会 forward evidence 上比较。只有达到配置的最小新增样本量后，程序才允许判断 rollback；如果胜率、平均收益、触发率、回撤等达到多项严重退化阈值，系统自动回到父版本并写入 `promotion.csv`。

回滚不会删除失败 release，也不会修改历史 ledger；系统进入受保护状态后继续收集 shadow evidence。

## Daily evidence and drift

每一个交易日都会立即贡献：

- 14:45 BUY/SELL/HOLD；
- tradable / DATA_HOLD 状态；
- ETF/策略线触发分布；
- 数据完整性和执行参考偏差。

这些信息可以立即用于日级漂移检测，但未平仓交易不能提前计入胜率或收益。只有 SELL 真正关闭此前记录的 BUY 后，才生成成熟的 `closed_trades.csv` 标签。

## Score

综合评分用于候选排序，不替代硬约束。收益、回撤、胜率、触发率、窗口稳定性、邻域稳定性和样本新鲜度必须分别保留可审计证据，不能用单一总分覆盖硬门槛。

## Audit and deployment status

2024+ 结果已被反复查看，只能称作后段审计窗口。真正的新证据从 live forward ledger 开始累计。

自动运行不依赖 GPT。部署顺序为：

1. 保持 `release_v2` 冻结；
2. 建立独立 C/S/D/R runtime copy；
3. 通过 Python 依赖、runtime、数据和 V2 baseline health checks；
4. 在 `SHADOW_ONLY` 收集真正新样本；
5. 只有显式切换到 `NORMAL` 后，程序才具备自动正式晋级权限；
6. 所有晋级仍必须通过上述硬 gate；
7. 自动 rollback 始终保留父 release 作为恢复点。

GitHub CI 必须同时通过 live optimizer 正式测试集、`verify_current_baseline.py` 和无 GPT/LLM runtime dependency 检查，实验分支才有资格进入合并评审。
