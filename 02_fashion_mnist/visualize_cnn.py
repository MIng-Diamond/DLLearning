"""
CNN 可视化 —— 看看卷积核到底学到了什么
==========================================

    cd 02_fashion_mnist
    python visualize_cnn.py

会生成三张图：

    cnn_filters.png      conv1 的 16 个 3×3 卷积核（模型学到的"模板"）
    cnn_features.png     一张图经过 conv1/conv2 后的特征图（每层在看什么）
    cnn_confidence.png   预测的置信度分布（分对 vs 分错）

═══════════════════════════════════════════════════════════════
 核心技巧：forward hook
═══════════════════════════════════════════════════════════════

PyTorch 的 nn.Module 支持「钩子」：在某一层的 forward 执行完后，
自动调用你注册的函数，把这一层的输入/输出交给你。

    layer.register_forward_hook(fn)
    # fn(module, input, output) 里可以保存 output

这样你不用改模型的 forward，就能抓到任意中间层的输出 ——
这是研究里分析模型最常用的手段。

本文件里抓的是：

    net[0] Conv2d(1,16)   → conv1 的原始输出
    net[1] ReLU           → 激活后的特征图       ← 我们画这个
    net[3] Conv2d(16,32)  → conv2 的原始输出
    net[4] ReLU           → 激活后的特征图       ← 和这个
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import get_loaders, CLASSES
from cnn import FashionCNN

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK HK",
    "WenQuanYi Zen Hei", "Droid Sans Fallback", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_MUTED = "#898781"
C_BLUE = "#2a78d6"
C_RED = "#e34948"


def load_model(path="cnn.pt", device="cpu"):
    model = FashionCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# ============================================================
# 图 1：卷积核
# ============================================================
def plot_filters(model, path="cnn_filters.png"):
    """把 conv1 的 16 个 3×3 卷积核画出来。

    conv1.weight 形状是 (16, 1, 3, 3)：
        16 个输出通道 × 1 个输入通道 × 3×3

    每个核就是一个「模板」—— 模型用它去和图像局部做点积，
    响应越强，说明这个位置越像这个模板。
    """
    w = model.net[0].weight.detach().cpu()      # (16, 1, 3, 3)
    n = w.shape[0]

    fig, axes = plt.subplots(4, 4, figsize=(7, 7.2))
    fig.patch.set_facecolor(C_SURFACE)
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis("off")
            continue
        k = w[i, 0].numpy()
        # 用发散色：蓝 = 负权重（抑制），红 = 正权重（激活）
        lim = max(abs(k.min()), abs(k.max()))
        ax.imshow(k, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_title(f"核 {i}", fontsize=8, color=C_INK2)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("conv1 学到的 16 个 3×3 卷积核", color=C_INK, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=130, facecolor=C_SURFACE)
    plt.close(fig)
    print(f"    卷积核已保存: {path}")


# ============================================================
# 图 2：特征图
# ============================================================
def plot_feature_maps(model, device, path="cnn_features.png"):
    """取一张测试图，看它在 conv1 / conv2 之后激活成什么样。

    用 forward hook 抓中间层输出，不用改模型。
    """
    _, test_loader = get_loaders(batch_size=1)
    xb, yb = next(iter(test_loader))
    xb = xb.to(device)

    # ── 注册两个 hook，把 ReLU 之后的输出存下来 ──
    feats = {}
    h1 = model.net[1].register_forward_hook(
        lambda m, i, o: feats.__setitem__("conv1", o.detach()))
    h2 = model.net[4].register_forward_hook(
        lambda m, i, o: feats.__setitem__("conv2", o.detach()))

    with torch.no_grad():
        logits = model(xb)
    h1.remove(); h2.remove()

    pred = logits.argmax(dim=1).item()
    f1 = feats["conv1"][0].cpu()      # (16, 28, 28)
    f2 = feats["conv2"][0].cpu()      # (32, 14, 14)

    fig = plt.figure(figsize=(13, 7))
    fig.patch.set_facecolor(C_SURFACE)

    # 原图
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.imshow(xb[0, 0].cpu(), cmap="gray")
    ax0.set_title(f"输入\n真实: {CLASSES[yb.item()]}  预测: {CLASSES[pred]}",
                  color=C_INK, fontsize=10)
    ax0.axis("off")

    # conv1 的 16 张特征图
    g1 = fig.add_gridspec(4, 8, left=0.28, right=0.63, top=0.88, bottom=0.08,
                          wspace=0.08, hspace=0.08)
    for i in range(16):
        a = fig.add_subplot(g1[i // 4, (i % 4) + 0])
        a.imshow(f1[i], cmap="viridis")
        a.axis("off")
    fig.text(0.45, 0.93, "conv1 特征图 16 张 (28×28)", ha="center",
             color=C_INK, fontsize=11)

    # conv2 的前 16 张
    g2 = fig.add_gridspec(4, 8, left=0.68, right=1.0, top=0.88, bottom=0.08,
                          wspace=0.08, hspace=0.08)
    for i in range(16):
        a = fig.add_subplot(g2[i // 4, (i % 4) + 0])
        a.imshow(f2[i], cmap="viridis")
        a.axis("off")
    fig.text(0.84, 0.93, "conv2 特征图（前 16 / 32）(14×14)", ha="center",
             color=C_INK, fontsize=11)

    plt.savefig(path, dpi=130, facecolor=C_SURFACE)
    plt.close(fig)
    print(f"    特征图已保存: {path}")


# ============================================================
# 图 3：置信度分布
# ============================================================
def plot_confidence(model, device, path="cnn_confidence.png"):
    """模型「有多确信」—— 分对的时候和分错的时候分别是什么分布。

    softmax 后的最大概率就是置信度。
    理想情况：分对时置信度接近 1，分错时也偏低（知道自己不确定）。
    """
    _, test_loader = get_loaders(batch_size=256)

    confs, corrects = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            probs = torch.softmax(model(xb), dim=1)
            conf, pred = probs.max(dim=1)
            confs.append(conf.cpu())
            corrects.append((pred == yb).cpu())

    conf = torch.cat(confs).numpy()
    correct = torch.cat(corrects).numpy()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor(C_SURFACE)
    ax.set_facecolor(C_SURFACE)

    bins = np.linspace(0, 1, 41)
    ax.hist(conf[correct], bins=bins, alpha=0.75, color=C_BLUE,
            label=f"预测正确 ({correct.sum()})")
    ax.hist(conf[~correct], bins=bins, alpha=0.75, color=C_RED,
            label=f"预测错误 ({(~correct).sum()})")

    ax.set_xlabel("置信度（softmax 最大概率）", color=C_INK2)
    ax.set_ylabel("样本数", color=C_INK2)
    ax.set_title(f"预测置信度分布      总体准确率 {correct.mean():.4f}",
                 color=C_INK)
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=C_MUTED)

    plt.tight_layout()
    plt.savefig(path, dpi=130, facecolor=C_SURFACE)
    plt.close(fig)

    print(f"    置信度图已保存: {path}")
    print(f"    平均置信度: 正确 {conf[correct].mean():.3f} | "
          f"错误 {conf[~correct].mean():.3f}")
    print(f"    错误样本占 {100*(~correct).mean():.2f}%")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("cnn.pt", device)

    plot_filters(model)
    plot_feature_maps(model, device)
    plot_confidence(model, device)
    print("\n完成。三张图都在 02_fashion_mnist/ 下。")


if __name__ == "__main__":
    main()
