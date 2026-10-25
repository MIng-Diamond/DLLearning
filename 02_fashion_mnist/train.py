"""
项目 · 第 2 节 —— 训练循环
============================

第 1 节你已经准备好了零件：
    data.py   get_loaders()  → 一批一批的 (xb, yb)
    model.py  FashionMLP     → logits (B, 10)

这一节把它们接起来，跑通四步循环：

    ① logits = model(xb)
    ② loss   = criterion(logits, yb)
    ③ loss.backward()
    ④ optimizer.step()

【CrossEntropyLoss】
    它内部 = softmax + 负对数似然。
    所以模型输出 logits（原始分数）就行，不要再 softmax。
    输入：logits (B, 10)，标签 yb (B,) 的类别下标（0~9），不是 one-hot。

【optimizer.zero_grad()】
    PyTorch 默认会【累加】梯度：这一步算出的梯度会加到上一步的 .grad 上。
    所以每个 batch 更新前必须清零，否则相当于把很多步的梯度混在一起。

【优化器在做什么】
    SGD 的核心就是：p ← p − lr * p.grad
    autograd 已经把每个参数的 p.grad 算好了，optimizer.step() 负责更新。
    momentum 只是让更新方向带一点惯性，帮助更快收敛。

【train / eval】
    model.train()：训练模式
    model.eval() ：评估模式
    本 MLP 没有 Dropout/BatchNorm，切换暂时没有实际影响，
    但一定要养成习惯 —— 以后写 CNN/Transformer 会救命。
    评估时用 with torch.no_grad(): 不建计算图，省内存也更快。

【设备】
    model 和 数据 必须在同一个设备上。
    model.to(device) 之后，每个 batch 也要 xb.to(device), yb.to(device)。
"""

import torch
import torch.nn as nn

from data import get_loaders
from model import FashionMLP


def get_device():
    """有 CUDA 就用 CUDA，否则 CPU。"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model, loader, criterion, device):
    """在 loader 上评估，返回 (平均损失, 准确率)。

    ⚠️ 平均损失不能用「每个 batch 的 loss 直接求平均」，
       因为最后一个 batch 可能不满 64 个样本。
       正确做法：loss 按 batch 大小加权求和，最后除以样本总数。

    步骤：
        1. model.eval()
        2. 初始化 total_loss = 0.0, correct = 0, total = 0
        3. with torch.no_grad():
               for xb, yb in loader:
                   xb, yb = xb.to(device), yb.to(device)
                   logits = model(xb)
                   loss = criterion(logits, yb)
                   total_loss += loss.item() * len(yb)
                   preds = logits.argmax(dim=1)
                   correct += (preds == yb).sum().item()
                   total += len(yb)
        4. 返回 total_loss / total, correct / total
    """
    # TODO ①
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(yb)
            y_pred = logits.argmax(dim=1)
            correct += (y_pred == yb).sum().item()
            total += len(yb)
    return total_loss / total, correct / total
    raise NotImplementedError("还没写 evaluate")

def train(model, train_loader, test_loader, epochs=20, lr=0.1,
          device=None, verbose=True):
    """训练 epochs 轮，返回 history 字典。

    history = {
        "train_loss": [...], "train_acc": [...],   # 每轮一个数
        "test_loss":  [...], "test_acc":  [...],
    }

    每一轮做两件事：
        1. 训练：model.train()，遍历 train_loader，对每个 batch 做四步更新
        2. 评估：调用 evaluate()，分别算训练集和测试集的 loss/acc

    优化器（用现成 API）：
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    损失函数：
        criterion = nn.CrossEntropyLoss()

    提示：
        - device 为 None 时，用 get_device()
        - 每个 batch 的顺序：zero_grad → forward → loss → backward → step
          （forward 和 loss 在 zero_grad 前后都行，关键是 backward 前清零）
        - history 里的数字用 float，方便打印和画图
        - verbose=True 时每轮打印一行，例如：
          epoch 1/5  train_loss=0.5231 train_acc=0.8123  test_loss=0.4321 test_acc=0.8456
    """
    # TODO ②
    if device is None:
        device = get_device()
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss":  [], "test_acc":  [],
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        train_loss, train_acc = evaluate(model, train_loader, criterion=criterion, device=device)
        test_loss, test_acc = evaluate(model, test_loader, criterion=criterion, device=device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        if verbose is True:
            print(f"epoch {epoch+1}/{epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
    return history
    raise NotImplementedError("还没写 train")


if __name__ == "__main__":
    device = get_device()
    print(f"使用设备: {device}")

    train_loader, test_loader = get_loaders(batch_size=64)
    model = FashionMLP(hidden=256).to(device)

    history = train(model, train_loader, test_loader,
                    epochs=20, lr=0.1, device=device)

    print(f"\n最终测试准确率: {history['test_acc'][-1]:.4f}")

    torch.save(model.state_dict(), "mlp.pt")
    print("模型已保存: mlp.pt")
