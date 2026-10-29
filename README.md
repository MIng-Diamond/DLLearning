# DLLearning

手写深度学习网络的学习仓库。

不调用现成模型，模型结构和训练循环自己写；数据加载、优化器、autograd 用 PyTorch 现成 API。

> A learning repository for hand-written deep learning networks. Models and training loops are written by hand; data loading, optimizers, and autograd use PyTorch APIs.

## 进度

| 阶段 | 内容 | 结果 |
|---|---|---|
| 0 | NumPy 基础、softmax、线性回归、梯度检验 | 白纸测试 5/5 |
| 1 | NumPy 手推 MLP 反向传播；PyTorch MLP | sin 拟合 loss 3.08 → 0.0031；Fashion-MNIST 86.88% |
| 项目零 | 加州房价线性回归 | R² = 0.6374，RMSE = 0.7048 |
| 2 | 手写 conv2d / maxpool；PyTorch CNN | 90.24% → 92.25% |
| 3 | RNN / GRU / LSTM；手写 Attention；Transformer | 序列反转 100%；Mini-GPT val loss 1.64 |

阶段 0–3 已完成。阶段 4（VAE、DDPM、PPO、Behavior Cloning）还没开始。

## 目录

```
DeepLearning/
├── 00_python_numpy/     NumPy、线性回归、梯度检验
├── 01_mlp/              NumPy 手写 MLP
├── 02_fashion_mnist/    MLP / CNN / 可视化
├── 03_housing/          加州房价完整项目
├── 03_rnn_attention/    RNN / LSTM / 梯度消失 / Attention
├── 04_transformer/      Transformer / Mini-GPT / Encoder-Decoder
├── demo/                交互 demo
├── notes/               踩坑记录
├── report/              阶段报告（只提交 Markdown 源文件）
└── ROADMAP.md           学习路线和进度
```

## 快速开始

```bash
conda activate learndl

# 加州房价
cd 03_housing && python run.py

# Fashion-MNIST
cd 02_fashion_mnist && python train.py

# Transformer 组件自检
cd 04_transformer && python debug_components.py

# 序列反转
cd 04_transformer && python lesson12.py

# tiny Shakespeare
cd 04_transformer && python lesson13_shakespeare.py

# 交互 demo
cd demo && python demo_text.py
cd demo && python demo_image.py --index 42
```

## 报告

- 中文：[report/report.md](report/report.md)
- English: [report/report_en.md](report/report_en.md)

`report/` 只提交这两份 Markdown；PDF、LaTeX、HTML 和构建脚本都是本地生成的，已加入 `.gitignore`。

报告里的实验图在各自目录下，例如 `02_fashion_mnist/confusion_matrix_improved.png`、`04_transformer/reversal_cross_attention.png`。这些图片不在 `.gitignore` 里，需要提交后报告才能正常显示。

## 环境

- conda 环境 `learndl`
- Python 3.11，torch 2.14.1+cu130，torchvision 0.29
- 数据和模型权重不提交（`.gitignore` 里的 `data/`、`*.pt`）

## 路线

详细计划和每课记录见 [ROADMAP.md](ROADMAP.md)。

## 说明

- 哪些是手写的、哪些是调用的，报告里分开写了。
- 序列反转是阶段 3 的过关任务：用 Encoder-Decoder Transformer，在没见过的 256 条序列上整句全对。
- 报告里的数字都来自实际训练；关键配置写在报告里，可以照着复现。
