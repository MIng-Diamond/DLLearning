"""
项目 · 第 1 节 —— 数据加载
=============================

真实项目和练习题的第一个区别：**数据不是 numpy 数组，是一堆文件。**

这里我们用 torchvision 自带的 Fashion-MNIST：
  - 6 万张训练图 + 1 万张测试图
  - 每张 28×28 灰度图，共 10 个类别（T恤、裤子、套头衫...）
  - 第一次运行会自动下载到 ./data（约 30MB）

【你可能会问】为什么要用 DataLoader，不直接用一个 numpy 大数组？
  三个原因，都很实际：
    1. **分批**：6 万张图不可能一次全塞进显存，要切成小批
    2. **打乱**：每轮训练前要重新洗牌，否则模型会记住顺序
    3. **预处理**：图片要转成张量、归一化，这些能在读取时顺手做掉
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 10 个类别的名字，按官方固定顺序。预测时要把索引翻译成这些名字。
CLASSES = [
    "T-shirt/top",  # 0
    "Trouser",      # 1
    "Pullover",     # 2
    "Dress",        # 3
    "Coat",         # 4
    "Sandal",       # 5
    "Shirt",        # 6
    "Sneaker",      # 7
    "Bag",          # 8
    "Ankle boot",   # 9
]


def get_loaders(batch_size=64, data_dir="./data"):
    """返回 (train_loader, test_loader)。

    三个步骤：

      1. 定义预处理 transform：
             transforms.ToTensor()
         它做两件事：
             - 把 PIL 图片转成 torch 张量，形状从 (28,28) 变成 (1,28,28)
             - 把像素值从 0~255 缩放到 0.0~1.0
         （归一化很重要：不缩放的话，梯度会被大数值带飞，训练不稳）

      2. 加载数据集：
             datasets.FashionMNIST(data_dir, train=True,  download=True, transform=transform)
             datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
         train=True 拿训练集，train=False 拿测试集。

      3. 包成 DataLoader：
             DataLoader(数据集, batch_size=batch_size, shuffle=?)
         训练集要 shuffle=True（每轮重新洗牌），测试集 shuffle=False。

    返回两个 DataLoader。
    """
    # TODO 1：写 transform
    transform = transforms.ToTensor()
    # TODO 2：加载 train_set 和 test_set
    trainset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    testset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    # TODO 3：包成 train_loader 和 test_loader 并返回
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return (train_loader, test_loader)
    raise NotImplementedError("还没写 get_loaders")


if __name__ == "__main__":
    # 写完可以直接跑这个文件，看看数据长什么样
    train_loader, test_loader = get_loaders()

    xb, yb = next(iter(train_loader))
    print(f"一个 batch 的图像: {xb.shape}   （B, C, H, W）")
    print(f"一个 batch 的标签: {yb.shape}")
    print(f"像素范围: {xb.min():.3f} ~ {xb.max():.3f}   （应该是 0~1）")
    print(f"标签样例: {yb[:8].tolist()}")
    print(f"对应的类别: {[CLASSES[i] for i in yb[:8].tolist()]}")
