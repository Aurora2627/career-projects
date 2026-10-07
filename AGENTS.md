# 用户的实验与源码要求

- 所有新增或修改的模型实验使用 Python 源码（.py）与 PyTorch。网络、训练、优化器、权重加载均以 PyTorch 实现，不使用 scikit-learn/joblib 训练或保存模型。
- 数据处理、绘图和运行工具也须保存 Python 源码；不要只保留 notebook 输出、命令片段或运行结果。
- 每次实验开始前保存当时的 Python 源码快照、配置、源代码和输入清单的 SHA256、环境信息；训练时保存日志，完成后保存 .pt 权重、指标与预测。
- 新建实验目录，禁止覆盖历史结果。用户授权清除旧实验时，必须先验证替代结果；清除范围仅限该指令涉及的实验产物，不扩大到原始数据或无关项目。
- source_history 只保存已替换的 Python 源码以便追溯，不作为当前训练入口。

## 当前日常开发环境

用户要求使用系统默认 Python。使用 `/usr/bin/python3`（3.9.6），并设置 `PYTHONPATH=/Users/meiying/Documents/Codex/2026-10-07/zhe/.system-python-packages`。系统用户包目录未取得写权限，因此这是系统解释器加载项目依赖，不是全局安装或虚拟环境。VS Code 配置位于 mm01/.vscode，说明为 mm01-content-classification/VSCODE_START.md。后续代码兼容 Python 3.9；历史实验源码快照与原环境记录保持不变。

VS Code 终端已实测 MPS 可用并通过 mps:0 float32 前向/反向/AdamW 更新，记录为 mm01-content-classification/reports/vscode-mps-check.json。Codex 运行进程中 MPS available=false 与 VS Code 不同；报告 GPU 能力时须区分执行环境。默认 train_torch.py 为 float32/AdamW，已在 VS Code 完成 MPS 特征提取和分类头训练（runs/vscode-mps-v1）；历史 float64/LBFGS 分类头仍在 CPU。所有后续训练、测试、预测必须从 VS Code 集成终端或调试器启动，禁止从 Codex 执行进程运行这些任务。源码读取、编辑可以使用文件工具。
