"""
PyTorch CNN 零件说明书 —— 每个零件单独跑一遍，看它对形状做了什么
====================================================================

    cd 02_fashion_mnist
    python pytorch_basics.py

不用理解全部，先看输出：
    输入 (2,1,28,28) 经过每个零件后，形状怎么变。
"""

import torch
import torch.nn as nn

print("=" * 62)
print("一、张量是什么")
print("=" * 62)

# 一张图 = 一个四维张量 (批量, 通道, 高, 宽)
x = torch.randn(2, 1, 28, 28)      # 2 张 1 通道 28×28 的图
print(f"x.shape = {tuple(x.shape)}")
print("          └┬┘ └┬┘ └─┬─┘ └─┬─┘")
print("           批量 通道  高    宽")
print()

print("=" * 62)
print("二、每个零件对形状做了什么")
print("=" * 62)

# ── Conv2d ──
conv = nn.Conv2d(
    in_channels=1,        # 输入有几个通道（灰度图=1，RGB=3）
    out_channels=16,      # 用几个卷积核 → 输出就有几个通道
    kernel_size=3,        # 卷积核 3×3
    padding=1,            # 四周补 1 圈 0 → 尺寸不变
)
y = conv(x)
print(f"Conv2d(1→16, k=3, p=1)  : {tuple(x.shape)} → {tuple(y.shape)}")
print("                          通道 1→16，高宽不变（padding=1 的功劳）")

# ── BatchNorm2d ──
bn = nn.BatchNorm2d(16)   # 参数 = 通道数
print(f"BatchNorm2d(16)          : {tuple(y.shape)} → {tuple(bn(y).shape)}   （形状不变）")

# ── ReLU ──
relu = nn.ReLU()
t = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
print(f"ReLU([-2,-0.5,0,0.5,2])  : {t.tolist()} → {relu(t).tolist()}   （负数变 0）")

# ── MaxPool2d ──
pool = nn.MaxPool2d(2)
print(f"MaxPool2d(2)             : {tuple(y.shape)} → {tuple(pool(y).shape)}   （高宽各减半）")

# ── Flatten ──
z = pool(y)
flat = nn.Flatten()
print(f"Flatten()                : {tuple(z.shape)} → {tuple(flat(z).shape)}")
print(f"                          16×14×14 = {16*14*14}")

# ── Linear ──
fc = nn.Linear(16 * 14 * 14, 10)
print(f"Linear({16*14*14}→10)          : {tuple(flat(z).shape)} → {tuple(fc(flat(z)).shape)}")
print("                          每个样本变成 10 个分数（10 个类别）")

# ── Dropout ──
drop = nn.Dropout(0.5)
print(f"Dropout(0.5)             : 形状不变，训练时随机把一半元素变成 0")

print()
print("=" * 62)
print("三、完整模型的形状流水线")
print("=" * 62)

from cnn import FashionCNN

model = FashionCNN()
x = torch.randn(2, 1, 28, 28)

print(f"输入            : {tuple(x.shape)}")
h = x
for i, layer in enumerate(model.net):
    h = layer(h)
    print(f"  net[{i}] {layer.__class__.__name__:12s} → {tuple(h.shape)}")

print()
print("=" * 62)
print("四、train() 和 eval() 的区别")
print("=" * 62)

# Dropout 只在 train 模式下生效
d = nn.Dropout(0.5)
t = torch.ones(10)

d.train()
print(f"train() 模式下 Dropout : {d(t).tolist()}")
print("                          ↑ 一半变成了 0（随机的）")

d.eval()
print(f"eval()  模式下 Dropout : {d(t).tolist()}")
print("                          ↑ 原样通过，不做丢弃")

# BatchNorm 也一样：train 用当前 batch 统计量，eval 用累积的滑动统计量
print()
print("这就是为什么评估前必须写 model.eval()。")
