# Repository Scope

`main` 只保存已经通过审计并准备作为正式基线的版本，以及必要的历史兼容层。当前 canonical baseline 为 `same_day_1445/release_v2`。

新的实验必须建立独立分支，例如 `opt/v3-*`；正式升级顺序为：实验分支 → 滚动审计 → 与 V2 冻结基线比较 → PR → 合并。不要直接覆盖 `release_v2`。

`file_manifest_sha256.csv` 的范围是“当前 canonical V2 + 当前审计工具与关键文档”，不是对整个历史 `strategies/` 树的永久全量清单。历史兼容层不参与当前 V2 完整性判定。

当前仓库为 Public。仅提交适合公开共享的研究代码、冻结结果和文档；其他材料应继续放在受控存储中。

## Live optimizer branch boundary

`opt/live-self-optimizer-v1` 是自优化系统实验分支。Phase 1 只提供 provider-agnostic 的核心引擎、append-only 账本、滚动评估、漂移检测、门控晋级、不可变 release 与 replay；它不宣称 GitHub 已经在自动抓取 14:45 实时分钟数据。

真实行情采集、四条正式信号适配、Windows 定时任务和网页运行面板属于后续阶段，必须在核心测试通过后单独接入。运行时新样本、分钟行情和可能包含大体积或敏感信息的证据不默认提交到 Public 仓库。
