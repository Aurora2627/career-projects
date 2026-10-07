# 版本恢复说明

这些提交由已保存的实验源码快照重建，提交日期是恢复日期，不能当作原始开发提交历史。

| 版本 | 保存的实验 | 内容 |
|---|---|---|
| v1 | pytorch-phase1-clip | PyTorch 单模态与图文融合基线 |
| v2 | pytorch-phase2-verified | 数据来源核验与严格少样本实验 |
| v3 | system-python-phase1-clip | 系统 Python 3.9 兼容与调试 |
| v4 | vscode-mps-v1 | VS Code 中 MPS/AdamW 四模型实验 |

最早的非 PyTorch 代码仅保存在 source_history，不伪造缺失的逐轮历史。权重、数据、缓存与本地依赖留在本机。version-history 保留公开的配置、指标和源码哈希；源代码版本通过 Git 提交恢复。

VS Code 设置包含本机绝对路径，未提交本机设置；可运行 scripts/configure_vscode.py 重新生成。所有模型运行仍从 VS Code 启动。

当前提交恢复快照：system-python-phase1-clip；上层项目规划与报告目录来自恢复时整理，不代表当时状态。
