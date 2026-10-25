"""
项目 · 第 3 节 —— 评估与可视化
================================

第 2 节你得到了一个 test_acc ≈ 0.842（3 轮）/ 会更高的模型。
但一个总准确率说明不了：

    - 哪些类别总被搞混？
    - 每类是不是都学得一样好？
    - 模型到底错在什么地方？

这一节把模型放到"显微镜"下：

    1. 加载 train.py 保存的 mlp.pt
    2. 在测试集上收集所有预测
    3. 混淆矩阵 10×10
    4. 每类准确率
    5. 把预测错的图片画出来

运行：
    cd 02_fashion_mnist
    python train.py        # 如果还没有 mlp.pt
    python eval_model.py
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import get_loaders, CLASSES
from model import FashionMLP
from cnn import FashionCNN

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK HK",
    "WenQuanYi Zen Hei", "Droid Sans Fallback", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def load_trained_mlp_model(path, device):
    """加载 train.py 保存的 state_dict，返回 eval 模式的模型。"""
    model = FashionMLP(hidden=256).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def load_trained_cnn_model(path, device):
    """加载 train.py 保存的 state_dict，返回 eval 模式的模型。"""
    model = FashionCNN().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def collect_predictions(model, loader, device):
    """返回 (preds, labels)，都是 numpy 数组，形状 (N,)。

    步骤：
        1. model.eval()
        2. preds_list, labels_list = [], []
        3. with torch.no_grad():
               for xb, yb in loader:
                   xb, yb = xb.to(device), yb.to(device)
                   logits = model(xb)
                   preds = logits.argmax(dim=1)
                   preds_list.append(preds.cpu())
                   labels_list.append(yb.cpu())
        4. torch.cat(...) 拼起来，再 .numpy()
    """
    # TODO ①
    model.eval()
    preds_list, labels_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            preds_list.append(preds.cpu())
            labels_list.append(yb.cpu())
    preds = torch.cat(preds_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    return (preds, labels)
    raise NotImplementedError("还没写 collect_predictions")


def confusion_matrix(preds, labels, num_classes=10):
    """返回 (num_classes, num_classes) 的 numpy 数组。

    行 = 真实类别，列 = 预测类别：
        cm[t, p] = 真实是 t、被预测成 p 的样本数

    提示：
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(labels, preds):
            cm[t, p] += 1
    """
    # TODO ②
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm
    raise NotImplementedError("还没写 confusion_matrix")


def per_class_accuracy(cm):
    """返回长度为 num_classes 的数组：第 i 类的准确率。

    第 i 类准确率 = cm[i, i] / cm[i, :].sum()

    提示：分母可能为 0（某类在测试集里没出现），用 np.divide 的 where 参数保护。
    """
    # TODO ③
    acc = np.zeros((cm.shape[0],))
    for i in range(cm.shape[0]):
        acc[i] = cm[i, i] / cm[i, :].sum()
    return acc
    raise NotImplementedError("还没写 per_class_accuracy")


def plot_confusion_matrix(cm, path="confusion_matrix.png"):
    """画 10×10 混淆矩阵热力图。（已写好）"""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASSES, fontsize=8)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    混淆矩阵已保存: {path}")


def plot_per_class_accuracy(acc, path="per_class_accuracy.png"):
    """画每类准确率柱状图。（已写好）"""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(CLASSES)), acc)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("准确率")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    每类准确率已保存: {path}")


def show_right_classified(model, loader, device, n=12, path="right_classified.png"):
    """把预测对的样本画出来（最多 n 张）。"""
    model.eval()
    shown = 0
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            right = (preds == yb).nonzero(as_tuple=True)[0]
            for idx in right.tolist():
                if shown >= n:
                    break
                img = xb[idx].cpu().squeeze(0)
                axes[shown].imshow(img, cmap="gray")
                axes[shown].set_title(
                    f"真实: {CLASSES[yb[idx].item()]}\n"
                    f"预测: {CLASSES[preds[idx].item()]}",
                    fontsize=8,
                )
                axes[shown].axis("off")
                shown += 1
            if shown >= n:
                break

    for ax in axes[shown:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    正例图片已保存: {path}")


def show_misclassified(model, loader, device, n=12, path="misclassified.png"):
    """把预测错的样本画出来（最多 n 张）。（已写好）"""
    model.eval()
    shown = 0
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            wrong = (preds != yb).nonzero(as_tuple=True)[0]
            for idx in wrong.tolist():
                if shown >= n:
                    break
                img = xb[idx].cpu().squeeze(0)
                axes[shown].imshow(img, cmap="gray")
                axes[shown].set_title(
                    f"真实: {CLASSES[yb[idx].item()]}\n"
                    f"预测: {CLASSES[preds[idx].item()]}",
                    fontsize=8,
                )
                axes[shown].axis("off")
                shown += 1
            if shown >= n:
                break

    for ax in axes[shown:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    错例图片已保存: {path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trained_cnn_model("cnn.pt", device)
    _, test_loader = get_loaders(batch_size=256)

    preds, labels = collect_predictions(model, test_loader, device)
    cm = confusion_matrix(preds, labels)
    acc = per_class_accuracy(cm)

    overall = cm.trace() / cm.sum()
    print(f"总体准确率: {overall:.4f}")
    print("每类准确率:")
    for name, a in zip(CLASSES, acc):
        print(f"  {name:12s} {a:.4f}")

    plot_confusion_matrix(cm)
    plot_per_class_accuracy(acc)
    show_misclassified(model, test_loader, device)
    show_right_classified(model, test_loader, device)
