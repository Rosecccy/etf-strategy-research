# Repository Scope

`main` 只保存已经通过审计并准备作为正式基线的版本，以及必要的历史兼容层。当前 canonical baseline 为 `same_day_1445/release_v2`。

新的实验必须建立独立分支，例如 `opt/v3-*`；正式升级顺序为：实验分支 → 滚动审计 → 与 V2 冻结基线比较 → PR → 合并。不要直接覆盖 `release_v2`。

`file_manifest_sha256.csv` 的范围是“当前 canonical V2 + 当前审计工具与关键文档”，不是对整个历史 `strategies/` 树的永久全量清单。历史兼容层不参与当前 V2 完整性判定。

当前仓库为 Public。仅提交适合公开共享的研究代码、冻结结果和文档；其他材料应继续放在受控存储中。
