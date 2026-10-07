"""Generate user-facing documentation from verified PyTorch rerun artifacts."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    verified=json.loads((ROOT/'reports/pytorch-migration-verification.json').read_text())
    names=['pytorch-phase1-resnet','pytorch-phase1-clip','pytorch-phase2-exploratory','pytorch-phase2-verified']
    reports={n:json.loads((ROOT/f'runs/{n}/metrics.json').read_text()) for n in names}
    lines=['# 前两阶段 Python/PyTorch 重跑与清理报告','','旧模型与旧评测输出已由 PyTorch 重跑结果替代。原始数据、图片、预训练编码器和固定划分保留；旧 Python 源码仅保存至 source_history/pre-pytorch-migration 供追溯。','','## 全量训练结果','','| 实验 | 训练/验证/测试 | 文本 Macro-F1 | 图像 Macro-F1 | 融合 Macro-F1 |','|---|---|---:|---:|---:|']
    for n,r in reports.items():
        full={v['mode']:v for v in r['results'] if v['shots'] is None}
        sizes='/'.join(str(r['sizes'][s]) for s in ['train','val','test'])
        lines.append(f"| {n} | {sizes} | {full['text']['test']['macro_f1']:.4f} | {full['image']['test']['macro_f1']:.4f} | {full['fusion']['test']['macro_f1']:.4f} |")
    lines+=['','## 重跑范围与实现','','第一阶段：TF-IDF/文本 + ResNet18、CLIP 文本/图像/融合、多数类对照。第二阶段：来源核验前与来源核验后两个固定清单，每类严格 2/4 条、5 个训练采样种子与全量训练，以及 1000 次分层配对测试 bootstrap。共保存 72 个训练分类器 .pt。','','网络使用 torch.nn.Linear，损失为加权交叉熵与 L2，优化器为 torch.optim.LBFGS。与前两阶段的逻辑回归任务保持一致，不将模型替换为完全不同结构。TF-IDF 由 Python/PyTorch 实现，训练集 SVD 改为 torch.linalg.svd 精确分解，标准化和指标也均由 PyTorch 实现。预训练编码器由 PyTorch 加载并重新前向提取特征；新特征缓存保存为 .pt。','','前后数值差异可能来自精确/近似 SVD、优化停止容差和浮点误差，不能将变化归因于框架本身。此前一次 AdamW/MLP 试跑也被清理，新报告仅记录本轮一致的线性分类协议。','','## 验证与可追溯性','','- 原三套清单逐字节 SHA256 一致，未重抽验证或测试集。','- 4 组实验的源码快照哈希全部通过；每组均保存当次 .py、环境、配置和输入清单副本。','- 每个 .pt 权重均重新加载，完整测试集 logits 与内存模型精确一致。','- 标准化仅拟合当次训练样本；C 仅由验证 Macro-F1 选择。','- 11 项测试通过，涵盖泄漏、感知哈希、TF-IDF、标准化、训练/权重加载与 AP。','','## 已清除的旧实验产物','']
    lines.extend('- '+p for p in verified['deleted_old_artifacts'])
    lines+=['','清理明细与验证记录：pytorch-migration-verification.json。输入/来源核验与源码历史不属于旧实验结果，因此保留。','','## 限制与下一步','','编码器仍冻结；未进行端到端视觉微调或 VLM LoRA。类别 4 仍极少；6 类 Macro-F1 受其波动影响。已复用旧测试集，不能声称全新盲测。近重复哈希不能排除所有语义重复，来源风格偏差仍需评估。下一步在固定训练/验证协议下实施可训练融合头或视觉层微调，并准备额外域外评估。']
    (ROOT/'reports/pytorch-rerun-report.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'reports/phase1-report.md').write_text('# 第一阶段 PyTorch 重跑\n\n当前结果：runs/pytorch-phase1-resnet 与 runs/pytorch-phase1-clip。\n\n详见 [完整重跑报告](pytorch-rerun-report.md)。旧实验结果已清除，不再作为当前报告引用。\n')
    (ROOT/'reports/phase2-report.md').write_text('# 第二阶段 PyTorch 重跑\n\n当前主要结果：runs/pytorch-phase2-verified；核验前探索对照：runs/pytorch-phase2-exploratory。\n\n详见 [完整重跑报告](pytorch-rerun-report.md)，少样本明细见对应 runs 目录的 summary.md。旧实验结果已清除。\n')
    (ROOT/'README.md').write_text('''# 图文内容分类与少样本适配

已完成前两阶段的 Python/PyTorch 重跑，当前所有模型训练、优化、TF-IDF/SVD、标准化与指标使用 Python/PyTorch；源码保存为 .py，模型与特征保存为 .pt。旧实验模型、评测与派生缓存已清除。

## 当前结果

- 第一阶段：runs/pytorch-phase1-resnet、runs/pytorch-phase1-clip。
- 第二阶段：runs/pytorch-phase2-exploratory、runs/pytorch-phase2-verified。
- 详细报告：[重跑报告](reports/pytorch-rerun-report.md)。

每组保存 source_snapshot、experiment-config.json、source-manifest.json、environment.json、input_manifests、training-log.jsonl、采样 ID、.pt 权重、预测、图表与指标。所有当次源码快照不可修改。

当前输入：data/processed/phase1、phase2-exploratory、phase2-verified；原始数据、图片和预训练编码器保留。source_history 仅保留替换前的 Python 源码，不作当前训练入口。

## 范围

公开数据集 6 类图文分类，覆盖内容治理 JD 的分类、数据质量、少样本与模型评测要求；不能直接描述为企业真实违规审核系统。当前编码器冻结，尚未进行 VLM SFT、LoRA 或端到端 CNN 调优。

## 使用

见 [运行说明](QUICKSTART.md)、[学习说明](LEARNING.md) 与上层 EXPERIMENT_RULES.md。源码入口：scripts/train_torch.py、scripts/run_baselines.py、scripts/run_phase2.py；模型实现：src/torch_models.py；实验实现：src/torch_runner.py。
''')
    (ROOT/'QUICKSTART.md').write_text('''# Python/PyTorch 运行说明

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
''')
    (ROOT/'reports/data-source.md').write_text('''# 数据来源与处理

原始社区数据来自 https://huggingface.co/datasets/ams-99/fakeddit_9k 。作者官方元数据通过 https://github.com/entitize/Fakeddit 链接取得。原文件校验记录为 source-files.json、official-source-files.json。

8646 条社区数据中，8562 条通过规范化标题与原始图片 URL 唯一匹配官方 ID，且标签、划分均一致；84 条无法唯一对应。来源核验后的清单排除这些记录。逐条与汇总记录为 provenance-matches.jsonl、provenance-audit.json。

第一阶段固定清单：308/65/64 条，精确文本/解码图像去重。第二阶段来源核验前与核验后均为 1007/305/304 条，加入 dHash 距离 ≤3 候选过滤。对应清单在 data/processed/phase1、phase2-exploratory、phase2-verified。本次 PyTorch 重跑保持清单 SHA256 不变，未重选测试集。

类别 4 稀缺，因此数据不能称作平衡数据，严格 8-shot 无法运行。Fakeddit 标签使用基于来源的远程监督，不表示每条样本都经人工事实鉴定；详见 https://arxiv.org/html/1911.03854 。尚需人工近重复检查与域外评估，当前不重新分发数据。
''')
    p=ROOT/'configs/execution.json';d=json.loads(p.read_text());d.update(status='pytorch_two_phase_rerun_verified',current_run='runs/pytorch-phase2-verified',model_backend='frozen_features_pytorch_softmax_lbfgs');p.write_text(json.dumps(d,indent=2))
    p=ROOT.parent/'README.md';s=p.read_text();start=s.index('## 当前完成情况');end=s.index('## 推荐启动顺序')
    s=s[:start]+'''## 当前完成情况

四个项目目录已建立。mm01 前两阶段已全部使用 Python/PyTorch 重跑，保存 Python 源码、72 个 .pt 分类器、配置、日志与评测，并清理旧实验模型、结果与派生缓存。包括来源核验、近重复过滤、单模态/融合基线与 5 种子严格 K-shot；VLM 微调尚未实施。其余项目尚未实现。

'''+s[end:];p.write_text(s)
    print('Updated current reports and run instructions')
if __name__=='__main__':main()
