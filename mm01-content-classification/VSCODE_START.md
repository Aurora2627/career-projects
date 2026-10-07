# 在 VS Code 使用系统默认 Python

解释器为 `/usr/bin/python3`（实际路径 `/Library/Developer/CommandLineTools/usr/bin/python3`），Python 3.9.6。不使用项目虚拟环境。兼容版本为 PyTorch 2.8.0、torchvision 0.23.0、Transformers 4.57.6，见 requirements-system-python.txt 与 requirements-system-python-lock.txt。

## 安装范围

系统用户包目录的写权限两次均未获授予，因此当前依赖装在工作区 `.system-python-packages`，通过 PYTHONPATH 提供给系统解释器。VS Code 的解释器路径、调试配置与新终端环境均已设置。它解决此项目的运行与调试，**不是向系统全局 site-packages 安装**。普通终端在不设置 PYTHONPATH 时仍不会自动加载这些项目依赖。

## 使用

1. 用 VS Code 打开本项目文件夹，或打开上层 `multimodal-project.code-workspace`。
2. 关闭原有集成终端，再新建终端，让新的依赖路径设置生效。
3. 如果状态栏已经是 Python 3.9.6，可直接运行；否则执行 “Python: Select Interpreter”，输入 `/usr/bin/python3`。
4. 打开“运行和调试”，选择 **01 系统 Python 与 PyTorch 检查**，按 F5。
5. 调试自己的文件选择 **02 调试当前 Python 文件**；先点击代码行号旁设置断点。
6. 日常 GPU 训练选择 07；03/04/05 保留历史 CPU/LBFGS 协议。每次输入新的实验目录名，避免覆盖已完成结果。

新版 VS Code 配置明确指定系统解释器。解释器选择及 launch.json 的 python 设置参见[官方说明](https://code.visualstudio.com/docs/python/environments)。兼容 Python 3.9 的 PyTorch 版本依据[PyTorch 包元数据](https://pypi.org/pypi/torch/2.8.0/json)。

## 在新建的 VS Code 终端检查

```bash
python3 --version
python3 -m pip --version
python3 -c "import sys,torch; print(sys.executable); print(torch.__version__)"
python3 scripts/check_system_environment.py
python3 -m unittest discover -s tests -v
```

新终端也提供 pip 命令。pip 的默认目标为项目依赖目录，不会尝试写入 macOS 系统路径。建议使用 `python3 -m pip`，确保 pip 与运行代码的解释器一致。

## 普通终端

从项目目录设置依赖路径后运行：

```bash
export PYTHONPATH="/Users/meiying/Documents/Codex/2026-10-07/zhe/.system-python-packages"
/usr/bin/python3 scripts/check_system_environment.py
/usr/bin/python3 scripts/run_baselines.py --backend clip --output runs/my-system-python-run
```

这些命令只影响当前终端会话，没有修改 shell 全局配置。重新安装依赖的源码入口为 scripts/install_system_dependencies.py，配置生成源码为 scripts/configure_vscode.py。

## 已验证

- 系统解释器真实 PyTorch 前向、反向和优化器更新通过。
- 项目 13 项测试通过（含 CPU/MPS 训练检查）；Python 3.9 不支持的 int.bit_count 已改为兼容实现。
- 三路 CLIP 基线已用系统解释器完成，结果在 runs/system-python-phase1-clip。
- ResNet 与 CLIP 原始图像推理、.pt 模型加载通过；与历史预测概率差异分别约 1.4e-6 和 9.4e-8。
- debugpy 协议实际验证断点、读取系统解释器变量、单步和继续完成，见 reports/system-python-debugger.json。此项验证使用调试适配器，未声称通过 VS Code 界面完成了 F5 操作。

保留原实验源码快照不变。旧实验使用 Python 3.12/PyTorch 2.14.1，当前系统环境为 Python 3.9/PyTorch 2.8.0；不要改写旧环境记录。

## VS Code 中的 Apple GPU 实测

已在 VS Code 的终端用系统 Python 3.9.6 / PyTorch 2.8.0 检测：MPS built/available 均为 true。Float32 模型和张量实际位于 mps:0，前向、反向与 AdamW 参数更新 3 步通过。结果为 reports/vscode-mps-check.json，源代码为 scripts/check_mps.py；调试配置新增“06 Apple GPU（MPS）检测”。

Codex 内执行进程的 available=false 与 VS Code 实测不同，不能据此前者判定 MacBook 不支持 MPS。当前默认入口 scripts/train_torch.py 使用 Float32/AdamW；特征提取和分类头训练均支持 MPS。历史 float64/LBFGS 入口仍使用 CPU。此检查只验证 GPU 可用与计算正确，没有测量相对 CPU 的加速比。

## 当前训练和断点调试

以后所有训练、测试、预测均从 VS Code 集成终端或调试器启动。选择“07 MPS 图文训练（四个模型）”按 F5，输入一个未使用的实验目录名。它使用系统 Python、MPS、验证集早停和四组对照。选“08 运行项目测试”按 F5 可运行 13 项测试。

建议在 src/accelerated_training.py 的 loss.backward() 或 optimizer.step() 行设置断点，观察 bx.shape、by、loss.item() 和 next(model.parameters()).device。F10 单步，F5 继续。保存的模型带有 CPU 权重副本，可以在 CPU 或 MPS 加载。

在 VS Code 终端也可运行：

```bash
/usr/bin/python3 scripts/train_torch.py --device mps --output runs/my-mps-01
/usr/bin/python3 scripts/verify_mps_prediction.py
```

实际四组实验为 runs/vscode-mps-v1；源代码快照包含 .vscode、配置和环境锁文件。图文线性融合测试 Macro-F1 0.6435，融合 MLP 0.5904。新协议仍使用既有社区子集与测试集，不能视为独立泛化验证。报告为 runs/vscode-mps-v1/summary.md。

GPU 训练代码的 debugpy 断点、读取 mps:0 参数、单步、继续完成已从 VS Code 终端验证，记录为 reports/vscode-mps-debugger.json。验证脚本使用调试适配器；与在界面点击 F5 的验证区别应保留。
