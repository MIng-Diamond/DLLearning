"""
CNN 升级版 —— 把准确率从 90% 推到 93%+
==========================================

直接运行：

    cd 02_fashion_mnist
    python cnn_improved.py

这个文件把 PyTorch 里最常用的「提分工具箱」全用上了。
跑完对比一下基础版（90.24%），你就知道每一招值多少。

═══════════════════════════════════════════════════════════════
 用了哪七招（按重要性排序）
═══════════════════════════════════════════════════════════════

1. 【BatchNorm】         Conv 之后、ReLU 之前。让每层输入分布稳定，训练更快更稳
2. 【数据增强】          随机翻转 + 随机裁剪。让模型见到更多"变体"，减少过拟合
3. 【更深的网络】        1→32→64→128，三层卷积
4. 【Dropout(0.5)】      最后分类头之前，随机丢弃一半神经元
5. 【Adam 优化器】       比 SGD 好调，lr=1e-3 通常就不错
6. 【余弦退火调度器】     学习率从大慢慢降到 0，后期更精细
7. 【AdaptiveAvgPool】   不管输入多大，自适应池化到 1×1，省参数

═══════════════════════════════════════════════════════════════
 关键的工程细节（这些比模型结构更重要）
═══════════════════════════════════════════════════════════════

⚠️ 训练集和测试集用【不同】的 transform：
      训练集：RandomHorizontalFlip + RandomCrop + ToTensor
      测试集：只 ToTensor
   测试集绝不能做随机增强，否则评估结果每次都不一样。

⚠️ BN 在 train/eval 模式下行为不同：
      train: 用当前 batch 的均值/方差
      eval : 用训练期间累积的滑动统计量
   所以 model.train() / model.eval() 必须切对。
"""

import os
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK HK",
    "WenQuanYi Zen Hei", "Droid Sans Fallback", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 模型
# ============================================================
class ImprovedCNN(nn.Module):
    """3 层卷积 + BN + 自适应池化 + Dropout 分类头"""

    def __init__(self, num_classes=10):
        super().__init__()

        # 特征提取：每一块都是 Conv → BN → ReLU → Pool
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 28 → 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 14 → 7

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 注意：这里【不用】AdaptiveAvgPool2d(1)
            # 28×28 的小图压成 1×1 会丢掉太多空间信息，flatten 更好
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),                          # (B,128,7,7) → (B,6272)
            nn.Dropout(0.5),
            nn.Linear(128 * 7 * 7, num_classes),   # (B,10) logits
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ============================================================
# 数据（训练集和测试集用不同 transform）
# ============================================================
def get_loaders(batch_size=128, data_dir="./data"):
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),         # 随机左右翻转
        transforms.RandomCrop(28, padding=4),      # 随机裁 28×28（先补到 36×36）
        transforms.ToTensor(),
    ])
    test_tf = transforms.ToTensor()                # ← 测试集只转张量

    train_set = datasets.FashionMNIST(data_dir, train=True,
                                       download=True, transform=train_tf)
    test_set = datasets.FashionMNIST(data_dir, train=False,
                                      download=True, transform=test_tf)

    # num_workers>0 让数据加载和 GPU 计算并行
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ============================================================
# 评估
# ============================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """注意 @torch.no_grad() 装饰器：整个函数都不建计算图"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += len(yb)
    return total_loss / total, correct / total


# ============================================================
# 训练
# ============================================================
def train(epochs=30, lr=1e-3, batch_size=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    train_loader, test_loader = get_loaders(batch_size)
    model = ImprovedCNN().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        scheduler.step()                            # 每轮结束调一次学习率

        tr_loss, tr_acc = evaluate(model, train_loader, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        print(f"epoch {epoch+1:2d}/{epochs}  lr={scheduler.get_last_lr()[0]:.2e}  "
              f"train {tr_loss:.4f}/{tr_acc:.4f}  test {te_loss:.4f}/{te_acc:.4f}")

    torch.save(model.state_dict(), "cnn_improved.pt")
    print(f"\n最终测试准确率: {history['test_acc'][-1]:.4f}")
    print("模型已保存: cnn_improved.pt")
    return history


def plot_curves(history, path="cnn_improved_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.grid(color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("epoch")

    axes[0].plot(history["train_loss"], color="#2a78d6", linewidth=2, label="训练")
    axes[0].plot(history["test_loss"], color="#eb6834", linewidth=2, label="测试")
    axes[0].set_ylabel("损失")
    axes[0].set_title("损失曲线")
    axes[0].legend(frameon=False)

    axes[1].plot(history["train_acc"], color="#2a78d6", linewidth=2, label="训练")
    axes[1].plot(history["test_acc"], color="#eb6834", linewidth=2, label="测试")
    axes[1].set_ylabel("准确率")
    axes[1].set_title("准确率曲线")
    axes[1].legend(frameon=False)

    plt.tight_layout()
    plt.savefig(path, dpi=120, facecolor="#fcfcfb")
    print(f"曲线已保存: {path}")


if __name__ == "__main__":
    h = train()
    plot_curves(h)
