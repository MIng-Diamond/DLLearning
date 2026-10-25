"""
项目 · 第 1 节 —— 手搭模型
=============================

【这是「PyTorch 手搭」的核心技能】

任何一个 PyTorch 模型都是这个骨架：

    class 你的模型名(nn.Module):
        def __init__(self):
            super().__init__()
            # 在这里【定义有哪些层】

        def forward(self, x):
            # 在这里【定义数据怎么流过去】
            return 输出

两条铁律：
  1. 必须继承 nn.Module
  2. 必须调用 super().__init__()（否则 torch 不知道怎么管你的层）

【为什么分开写 __init__ 和 forward？】
  __init__ 只跑一次，负责「准备零件」；
  forward  每次前向都跑，负责「装配」。
  分开之后，torch 才能自动帮你做两件事：
    - model.parameters()  自动收集所有层的参数（优化器要用）
    - model.to("cuda")    一句话把整个模型搬到 GPU

【本项目用的结构】

    输入 (B, 1, 28, 28)
        ↓  展平 flatten
       (B, 784)
        ↓  nn.Linear(784, 256) + ReLU
       (B, 256)
        ↓  nn.Linear(256, 10)
       (B, 10)   ← logits：10 个类别的原始分数（还没 softmax）

⚠️ 注意最后【不要】加 softmax！
   因为交叉熵损失 nn.CrossEntropyLoss 内部已经包含了 softmax，
   再加一次就重复了。这是 PyTorch 新手常踩的坑。
"""

import torch
import torch.nn as nn


class FashionMLP(nn.Module):
    """两层 MLP：784 → 256 → 10"""

    def __init__(self, hidden=256):
        super().__init__()
        # TODO 1：定义两层全连接 + 一个 ReLU
        #     self.fc1  = nn.Linear(784, hidden)
        #     self.relu = nn.ReLU()
        #     self.fc2  = nn.Linear(hidden, 10)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )
        #raise NotImplementedError("还没定义层")

    def forward(self, x):
        """x: (B, 1, 28, 28)  →  返回 (B, 10) 的 logits

        步骤：
          1. 展平：把 (B, 1, 28, 28) 变成 (B, 784)
             提示：x.view(x.size(0), -1)   或   x.flatten(1)
             （-1 的意思是「这一维你自己算」，它会自动算出 784）
          2. 过 fc1
          3. 过 relu
          4. 过 fc2
        """
        # TODO 2
        return self.net(x)
        raise NotImplementedError("还没写 forward")


if __name__ == "__main__":
    model = FashionMLP()
    x = torch.randn(8, 1, 28, 28)          # 假装一个 batch 有 8 张图
    out = model(x)
    print(f"输入: {x.shape}  →  输出: {out.shape}   （应该是 (8, 10)）")
    n = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {n:,}")
    print()
    print("每一层的形状：")
    for name, p in model.named_parameters():
        print(f"  {name:12s} {tuple(p.shape)}")
