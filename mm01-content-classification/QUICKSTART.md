**更新：日常开发已配置系统默认 Python，见 [VS Code 启动说明](VSCODE_START.md)。下方保留原实验环境的复现命令。**

# Python/PyTorch 运行说明

从工作区 /Users/meiying/Documents/Codex/2026-10-07/zhe 运行。独立环境为 .venvs/mm01（Python 3.12），当前 Mac 进程使用 CPU。预训练 CLIP 与 ResNet 权重已本地缓存，CLIP 使用本地加载。

## 第一阶段

```bash
.venvs/mm01/bin/python outputs/career-projects/mm01-content-classification/scripts/run_baselines.py --backend resnet --output runs/my-phase1-resnet
.venvs/mm01/bin/python outputs/career-projects/mm01-content-classification/scripts/run_baselines.py --backend clip --output runs/my-phase1-clip
```

## 第二阶段

```bash
.venvs/mm01/bin/python outputs/career-projects/mm01-content-classification/scripts/run_phase2.py --output runs/my-phase2
.venvs/mm01/bin/python outputs/career-projects/mm01-content-classification/scripts/summarize_phase2.py --run runs/my-phase2
```

默认使用来源核验后的清单；加入 --data data/processed/phase2-exploratory 可重跑来源核验前的探索版本。所有训练目录必须尚不存在。默认复用同一输入对应的 PyTorch .pt 特征缓存。

## 单条预测

```bash
.venvs/mm01/bin/python outputs/career-projects/mm01-content-classification/scripts/predict.py --artifact runs/pytorch-phase2-verified/full-fusion.pt --image /你的图片绝对路径/example.jpg --text "图片对应的文本"
```

输出数值类别和概率。标签保持数据集定义，不能当作独立事实核验结果。只加载本项目可信的 .pt 权重，加载使用 weights_only=True。

## 图表与测试

```bash
.venvs/mm01/bin/python outputs/career-projects/mm01-content-classification/scripts/summarize_run.py runs/pytorch-phase2-verified
.venvs/mm01/bin/python -m unittest discover -s outputs/career-projects/mm01-content-classification/tests -v
```

每次实验自动保存 Python 源码快照、输入清单副本、环境、配置、哈希、采样 ID、优化日志和 .pt 模型。源代码历史目录仅供追溯；当前训练入口不依赖 scikit-learn/joblib。


## 当前 VS Code / MPS 入口

以后运行统一在 VS Code。打开运行和调试，选择“07 MPS 图文训练（四个模型）”，按 F5，输入新的实验目录名。默认 train_torch.py 已改为 Float32/AdamW，编码器与分类器均使用可用 GPU；旧实验复现必须指定 --optimizer lbfgs，仍使用 CPU。详情见 VSCODE_START.md。

最新实际运行：runs/vscode-mps-v1。四组测试 Macro-F1：文本 0.5191、图像 0.5082、图文线性融合 0.6435、融合 MLP 0.5904。源代码、VS Code 配置、输入哈希、日志与四份 .pt 权重均保存。来源为既有社区子集，单种子且重复使用测试集；这是探索结果。
