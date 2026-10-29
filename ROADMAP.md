# 手写深度学习网络 · 学习路线

> 目标：具备为**具身智能课题组**工作所需的网络实现能力
> 标准：**PyTorch 手搭模型** —— 用 `nn.Parameter` + autograd，自己写 forward 和训练循环，不抄现成模型
> 起点：会基本 Python 语法，未接触过 NumPy

---

## 每个网络的「六步法」（不要跳步）

1. **推** —— 白纸推导前向公式 + 梯度公式
2. **检** —— 用 2×3 之类的小矩阵手算一遍数值
3. **写** —— 实现，全程用 `assert` 标注张量形状
4. **验** —— 数值梯度检验（NumPy 阶段）／ 与参考实现对比（PyTorch 阶段）
5. **训** —— 玩具任务上训到收敛；先试「用 20 个样本过拟合」
6. **比** —— 与标准实现对比参数量、loss 曲线、精度

**铁律**：形状分析是主要调试手段；验证没过不要往下走；不要抄答案。

---

## 阶段与过关标准

### 阶段 0 · NumPy 过渡（约 1 周）— 目的是「看见」梯度
- `00_python_numpy/` — 形状、索引、axis、广播、矩阵乘法
- 手写 softmax（数值稳定版）
- 手写线性回归 + 梯度下降
- **过关**：能不看资料写出 softmax 和线性回归训练循环，loss 单调下降

### 阶段 1 · MLP（约 1 周）— 反向传播的骨架
- 两层 MLP：NumPy 手推反向传播（只此一次，为理解）
- 然后 PyTorch 手搭：手写 `nn.Module`、手写 train loop、手写 SGD/Adam
- **过关**：MNIST ≥ 97%（PyTorch 版）；NumPy 版梯度检验相对误差 < 1e-6

### 阶段 2 · CNN / ResNet（约 1.5 周）
- 手搭 Conv2d 等价实现、BatchNorm、残差块
- **过关**：MNIST ≥ 99%；能自己解释 BN 训练/推理的差异

### 阶段 3 · 序列与注意力（约 2 周）— 现代 DL 分水岭
- RNN cell → LSTM / GRU（手搭，理解门控）
- Scaled Dot-Product Attention → 多头注意力 → Transformer Block
- **过关**：Transformer 在「序列反转」任务 100%；训一个字符级 Mini-GPT

### 阶段 4 · 具身智能直系（约 3 周）
- VAE（重参数化）→ **DDPM 扩散模型**（Diffusion Policy 的地基）
- 强化学习：DQN → **PPO**
- 模仿学习：**Behavior Cloning** 最小闭环
- **过关**：PPO 解 CartPole；DDPM 生成可辨认的手写数字；BC 复现专家轨迹

### 选学（前三阶段吃透之前不要碰）
GNN、NeRF、Mamba、世界模型

---

## 环境（`learndl` 待自己动手创建 ⬜）

用 conda。两个环境，**分工明确**：

| 环境 | 用途 | 包里有什么 |
|---|---|---|
| **`learndl`** | **日常学习沙箱**（阶段 0~3） | Python 3.10.22、numpy 2.2.6、matplotlib 3.10.9、jupyterlab 4.6.3 |
| `embodied` | 课题组环境，阶段 4 需要 MuJoCo 时用 | Python 3.10.21、torch 2.14.0+cu130、mujoco 3.13.0、gymnasium 1.3.0 |

```bash
conda activate learndl         # 每次开始学习第一件事
conda deactivate               # 退出
```

- 显卡：RTX 3060 Laptop，**5.7 GB 可用显存**（训大模型时 batch size 要调小）
- VSCode 解释器选：`~/miniforge3/envs/learndl/bin/python`
- torch 等到**阶段 1** 再装进 `learndl`

## 进度记录

| 日期 | 内容 | 状态 |
|---|---|---|
| 2026-10-11 | 搭建仓库骨架、制定路线 | ✅ |
| 2026-10-11 | 亲手创建 conda 环境 `learndl`（Python 3.10 + numpy + matplotlib + jupyter + torch+cu130） | ✅ |
| 2026-10-11 | 第 01 课 NumPy 基础（形状/索引/axis/广播/矩阵乘法/softmax） | ✅ 6/6 通过 |
| 2026-10-11 | 第 02 课 手写线性回归 + 梯度下降（w=2.989, b=1.964） | ✅ 4/4 通过 |
| 2026-10-11 | 第 03 课 多变量线性回归（向量版梯度 `dw = (2/N)·Xᵀ·d`） | ✅ 4/4 通过 |
| 2026-10-11 | 第 04 课 PyTorch 对照（autograd、zero_grad、GPU） | ✅ 3/3 通过 |
| 2026-10-11 | **阶段 0 通关测试**（白纸默写：softmax + 线性回归训练循环） | ✅ 5/5 通过 |
| 2026-10-17 | 第 05 课 手写 MLP（NumPy 版，拟合 sin(x)）—— 原为选修，已收尾 | ✅ 4/4 通过（梯度检验通过；loss 从 ~1 降至 ~3e-3，优于最优线性模型 5 倍以上） |
| 2026-10-18 | 阶段 1 正式过关（MNIST ≥ 97%）—— 项目一已完成（Fashion-MNIST MLP 86.88%）；MNIST 补测暂缓，先进入阶段 2 | ⏸ 待补 |
| 2026-10-25 | 第 11 课组件全部手写完成：位置编码 / 多头注意力 / FFN / 残差 Block / MiniGPT / 训练循环 / 自回归生成；`debug_components.py` 7/7 通过 | ✅ |
| 2026-10-28 | 第 11 课完整训练跑通：2000 步 loss 从 4.0 降到 0.05，能生成 Alice 文本；进入第 12 课（Encoder-Decoder + 交叉注意力） | ✅ |
| 2026-10-28 | **阶段 3 通关**：第 12 课 Encoder-Decoder + 交叉注意力 5/5 通过；序列反转在没见过的 256 条序列上 100% 全对 | ✅ |
| 2026-10-28 | 第 13 课完成：tiny Shakespeare（111 万字符）训练 4.77M 字符级 Mini-GPT，6000 步；train 1.04 / val 1.64 / perplexity 5.13 | ✅ |
| 2026-10-28 | 交互 demo 完成：`demo/demo_text.py`（Mini-GPT 生成）+ `demo/demo_image.py`（Fashion-MNIST top-3） | ✅ |
| 2026-10-28 | 阶段报告完成：中文版 `report/report.md` / `.tex` / `.pdf`，英文版 `report/report_en.md` / `.tex` / `.pdf`（各 14 页，11 张实验图） | ✅ |

### 阶段 2 · CNN（`02_cnn/`，先手写，再用 PyTorch 做项目）

| 课 | 内容 | 状态 |
|---|---|---|
| 06 | 手写 2D 卷积 + 多通道 + 最大池化，与 PyTorch 逐位对照 | ✅ 4/4 |
| 07 | 手写卷积/池化反向传播 | ⏭️ 跳过（用户选择直接学 PyTorch 实战） |
| → | PyTorch CNN 做 Fashion-MNIST | ✅ 基础版 90.24% / 升级版 92.25% |
| → | CNN 可视化（卷积核 / 特征图 / 置信度 / 混淆矩阵 / 错例） | ✅ |

### 阶段 3 · 序列与注意力 ✅ 已通关

| 课 | 目录 | 内容 | 状态 |
|---|---|---|---|
| 08 | `03_rnn_attention/` | RNN 原理 + 手写 cell 前向 + LSTM 正弦波预测 | ✅ 2/2 |
| 09 | `03_rnn_attention/` | 梯度消失实验：RNN / GRU / LSTM(bias=1) 对比 | ✅ 2/2 |
| 10 | `03_rnn_attention/` | 缩放点积注意力 + 多头注意力（手写，对照 PyTorch） | ✅ 2/2 |
| 11 | `04_transformer/` | **Transformer Block + Mini-GPT**（位置编码 / 残差 / LayerNorm / FFN / 生成） | ✅ 完整训练完成（loss 4.0 → 0.05） |
| 12 | `04_transformer/` | **Encoder-Decoder + 交叉注意力**（序列反转，阶段 3 过关任务） | ✅ 5/5（泛化 100%） |
| 13 | `04_transformer/` | **真实语料训练 Mini-GPT**（tiny Shakespeare，4.77M 参数） | ✅ train 1.04 / val 1.64 / ppl 5.13 |

**阶段 3 过关任务 = 第 12 课**：序列反转（Encoder-Decoder Transformer）—— ✅ 已完成，没见过的 256 条序列整句全对 100%。

**目录划分原则**：`03_rnn_attention` = 序列建模（RNN → LSTM → Attention，Attention 在这里是"解决 RNN 长程依赖"的方案）；`04_transformer` = 把 Attention 组装成完整架构。

### 项目一 · Fashion-MNIST 分类器（PyTorch 手搭）

**API 约定**：数据加载/优化器/autograd 用现成 API；**模型结构和训练循环必须手写**。

| 日期 | 内容 | 状态 |
|---|---|---|
| 2026-10-17 | 第 1 节 数据管线 + 手搭 `nn.Module` | ✅ check1 2/2 |
| 2026-10-18 | 第 2 节 训练循环 + 评估（交叉熵 / 优化器 / 准确率） | ✅ check2 2/2（3 轮 test_acc=0.842） |
| 2026-10-18 | 第 3 节 评估与可视化（混淆矩阵 / 每类准确率 / 错例 / 正确样本） | ✅ check3 PASS（总体 86.88%） |
| | **项目一当前成绩** | **test_acc = 86.88%（MLP，203,530 参数，20 轮）** |
| 2026-10-18 | 尝试增加训练轮数（20 epochs）提升 MLP —— 判断提升有限，决定转入 CNN | ⏭️ 跳过 |

### 项目二 · CNN / ResNet（PyTorch 手搭）

**目标**：用卷积结构把 Fashion-MNIST 从 84% 推到 ≥ 92%；理解局部感受野、权重共享、池化、BatchNorm、残差连接。

| 日期 | 内容 | 状态 |
|---|---|---|
| 2026-10-18 | 第 1 节 手搭小型 CNN（Conv → ReLU → Pool ×2 → FC） | ⬜ 待做 |
| | 第 2 节 手写 Conv2d 等价实现（理解卷积数学） | ⬜ 待做 |
| | 第 3 节 BatchNorm + 残差块 | ⬜ 待做 |
| | **目标** | **Fashion-MNIST ≥ 92%；MNIST ≥ 99%（阶段 2 原标准）** |

### 项目零 · 加州房价线性回归（已完成）

| 日期 | 内容 | 结果 |
|---|---|---|
| 2026-10-11 | 第 1 节 数据：加载 / 标准化 / 划分（防数据泄漏） | ✅ 2/2 |
| 2026-10-17 | 第 2 节 模型 / 训练 / RMSE / R² / 预测 / 诊断图 | ✅ 3/3 |
| | **最终成绩** | **R²(测试)=0.6374，RMSE=0.70（约7万美元）** |
