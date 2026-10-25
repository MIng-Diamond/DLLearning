"""
项目二 · 第 1 节 —— 手搭第一个 CNN
===================================

MLP 把图像 flatten 成 784 个数，丢掉了“谁挨着谁”。
CNN 用卷积核看局部，用池化压缩尺寸。

本课架构：

    输入 (B, 1, 28, 28)
      Conv2d(1, 16, 3, padding=1)   → (B, 16, 28, 28)
      ReLU
      MaxPool2d(2)                  → (B, 16, 14, 14)
      Conv2d(16, 32, 3, padding=1)  → (B, 32, 14, 14)
      ReLU
      MaxPool2d(2)                  → (B, 32, 7, 7)
      Flatten                       → (B, 1568)
      Linear(1568, 10)              → (B, 10)  logits

参数量：
    Conv1: 16×1×3×3 + 16 = 160
    Conv2: 32×16×3×3 + 32 = 4,640
    FC:    1568×10 + 10 = 15,690
    合计: 20,490   （MLP 是 203,530，参数少约 10 倍）

为什么 CNN 更强：
    1. 局部感受野：3×3 只看邻域，先提取边缘/角点/纹理
    2. 权重共享：同一个卷积核扫全图，参数量少、平移不变
    3. 池化：尺寸减半，保留主要特征，扩大感受野

⚠️ 最后仍然不要加 softmax：CrossEntropyLoss 内部会做。
"""

import torch
import torch.nn as nn


class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO ①：定义层
        #   self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        #   self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        #   self.pool  = nn.MaxPool2d(2)
        #   self.fc    = nn.Linear(32 * 7 * 7, 10)
        self.net = nn.Sequential(nn.Conv2d(1, 16, kernel_size = 3, padding = 1),
                                 nn.BatchNorm2d(16),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2),
                                 nn.Conv2d(16, 32, kernel_size = 3, padding = 1),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2),
                                 nn.Flatten(),
                                 nn.Dropout(0.5),
                                 nn.Linear(32 * 7 * 7, 10))

        #raise NotImplementedError("还没定义层")

    def forward(self, x):
        """x: (B,1,28,28) → (B,10)"""
        # TODO ②：conv1 → relu → pool → conv2 → relu → pool → flatten → fc
        # 提示：
        #   x = self.pool(torch.relu(self.conv1(x)))
        #   ...
        return self.net(x)
        raise NotImplementedError("还没写 forward")


if __name__ == "__main__":
    from data import get_loaders
    from train import train, get_device

    device = get_device()
    print(f"使用设备: {device}")

    train_loader, test_loader = get_loaders(batch_size=64)
    model = FashionCNN().to(device)

    history = train(model, train_loader, test_loader,
                    epochs=8, lr=0.02, device=device)

    print(f"\n最终测试准确率: {history['test_acc'][-1]:.4f}")
    torch.save(model.state_dict(), "cnn.pt")
    print("模型已保存: cnn.pt")
