# 本机运行评估

检查日期：2026-10-07。本机 MacBook Air，Apple M5，16 GB 统一内存，macOS 26.6，工作目录磁盘剩余约 397 GiB。系统 Python 3.9.6；应用随附 Python 3.12.14（应用运行时路径可能随版本变化）。

当前检测环境未安装 torch、transformers、mlx、fastapi。不能据此确认 MPS 已可运行。

## 本地策略

- Agent：CPU 本地后端 + 远程模型 API；不依赖 CUDA。
- CNN：PyTorch MPS 优先、CPU 回退，限制 batch 和输入尺寸。
- CLIP：先用小型编码器，缓存特征后训练分类头，避免全参数训练。
- VLM 推理：MLX-VLM 小型量化模型或 API；仍需验证具体模型的兼容性、图像 token 和峰值内存。16 GB 是系统共享内存，不能全部作为模型预算。
- VLM LoRA/SFT：小型 MLX 实验理论可尝试，未实测；正式结果优先外部 Linux GPU/PyTorch。
- DeepSpeed/CUDA 分布式、常见 CUDA FlashAttention 路线放在外部 GPU；不把 Docker 当成 Mac 获得 NVIDIA CUDA 的途径。vLLM 如需生产路线优先外部 Linux 服务器，本机选 MLX/API。

## 后续需要验证

安装独立虚拟环境后探测 MPS；跑真实矩阵运算和小 batch 前向/反向；逐项目测量耗时与内存。权重可加载不等于训练可行。

## 参考

- https://docs.pytorch.org/docs/2.14/notes/mps.html
- https://github.com/Blaizzy/mlx-vlm
- https://docs.vllm.ai/en/latest/getting_started/installation/
