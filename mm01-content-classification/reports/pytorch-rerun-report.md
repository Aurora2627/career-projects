# 前两阶段 Python/PyTorch 重跑与清理报告

旧模型与旧评测输出已由 PyTorch 重跑结果替代。原始数据、图片、预训练编码器和固定划分保留；旧 Python 源码仅保存至 source_history/pre-pytorch-migration 供追溯。

## 全量训练结果

| 实验 | 训练/验证/测试 | 文本 Macro-F1 | 图像 Macro-F1 | 融合 Macro-F1 |
|---|---|---:|---:|---:|
| pytorch-phase1-resnet | 308/65/64 | 0.2468 | 0.4027 | 0.3961 |
| pytorch-phase1-clip | 308/65/64 | 0.4698 | 0.5272 | 0.5256 |
| pytorch-phase2-exploratory | 1007/305/304 | 0.5230 | 0.5135 | 0.6066 |
| pytorch-phase2-verified | 1007/305/304 | 0.5208 | 0.4992 | 0.5885 |

## 重跑范围与实现

第一阶段：TF-IDF/文本 + ResNet18、CLIP 文本/图像/融合、多数类对照。第二阶段：来源核验前与来源核验后两个固定清单，每类严格 2/4 条、5 个训练采样种子与全量训练，以及 1000 次分层配对测试 bootstrap。共保存 72 个训练分类器 .pt。

网络使用 torch.nn.Linear，损失为加权交叉熵与 L2，优化器为 torch.optim.LBFGS。与前两阶段的逻辑回归任务保持一致，不将模型替换为完全不同结构。TF-IDF 由 Python/PyTorch 实现，训练集 SVD 改为 torch.linalg.svd 精确分解，标准化和指标也均由 PyTorch 实现。预训练编码器由 PyTorch 加载并重新前向提取特征；新特征缓存保存为 .pt。

前后数值差异可能来自精确/近似 SVD、优化停止容差和浮点误差，不能将变化归因于框架本身。此前一次 AdamW/MLP 试跑也被清理，新报告仅记录本轮一致的线性分类协议。

## 验证与可追溯性

- 原三套清单逐字节 SHA256 一致，未重抽验证或测试集。
- 4 组实验的源码快照哈希全部通过；每组均保存当次 .py、环境、配置和输入清单副本。
- 每个 .pt 权重均重新加载，完整测试集 logits 与内存模型精确一致。
- 标准化仅拟合当次训练样本；C 仅由验证 Macro-F1 选择。
- 11 项测试通过，涵盖泄漏、感知哈希、TF-IDF、标准化、训练/权重加载与 AP。

## 已清除的旧实验产物

- runs/baseline-v1
- runs/baseline-v2
- runs/clip-v1
- runs/clip-v2
- runs/phase2
- runs/phase2-incomplete
- runs/phase2-verified
- runs/torch-v1
- data/cache/features-0e0183b099ec192f.npz
- data/cache/features-64da199a4b5f6562.npz
- data/cache/phase2-clip-324b901b8b7c76e7.npz
- data/cache/phase2-clip-af267278836bbb2f.npz
- data/processed/phase2
- reports/phase2-verification.json
- reports/verification.json
- reports/runtime.json

清理明细与验证记录：pytorch-migration-verification.json。输入/来源核验与源码历史不属于旧实验结果，因此保留。

## 限制与下一步

编码器仍冻结；未进行端到端视觉微调或 VLM LoRA。类别 4 仍极少；6 类 Macro-F1 受其波动影响。已复用旧测试集，不能声称全新盲测。近重复哈希不能排除所有语义重复，来源风格偏差仍需评估。下一步在固定训练/验证协议下实施可训练融合头或视觉层微调，并准备额外域外评估。
