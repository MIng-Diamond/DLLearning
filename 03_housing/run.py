"""
项目一 · 第 2 节 —— 一键跑通全流程
=====================================

    cd ~/projects/learn/DeepLearning/03_housing
    python run.py

会依次做：加载 → 训练 → 评估 → 预测示例 → 出四张诊断图。
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 让 matplotlib 能显示中文（否则图上的中文会变成方块 □□□）
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK HK",
    "WenQuanYi Zen Hei", "Droid Sans Fallback", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

from data import load_data, split_and_standardize, FEATURE_NAMES, TARGET_NAME
from model import predict, mse_loss, train
from evaluate import rmse, r2_score, residual_report
from predict import predict_raw


# ── 配色（已通过色盲可辨性校验的调色板）────────────────────
C_SURFACE = "#fcfcfb"   # 画布
C_INK     = "#0b0b0b"   # 主文字
C_INK2    = "#52514e"   # 次要文字
C_MUTED   = "#898781"   # 坐标轴刻度
C_GRID    = "#e1e0d9"   # 网格（要淡，不能抢戏）
C_AXIS    = "#c3c2b7"   # 轴线
C_POS     = "#2a78d6"   # 蓝 —— 正权重
C_NEG     = "#e34948"   # 红 —— 负权重

FEATURE_CN = {
    "MedInc": "收入中位数", "HouseAge": "房龄", "AveRooms": "平均房间数",
    "AveBedrms": "平均卧室数", "Population": "人口", "AveOccup": "每户人数",
    "Latitude": "纬度", "Longitude": "经度",
}


def style_ax(ax):
    """统一的坐标轴样式：网格退到数据后面，去掉多余边框。"""
    ax.set_facecolor(C_SURFACE)
    ax.grid(True, color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_AXIS)
    ax.tick_params(colors=C_MUTED, labelsize=9)
    ax.title.set_color(C_INK)
    ax.xaxis.label.set_color(C_INK2)
    ax.yaxis.label.set_color(C_INK2)


def make_figures(w, b, losses, X_tr, y_tr, X_te, y_te, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    fig.patch.set_facecolor(C_SURFACE)

    # ── 图1：训练损失曲线（单序列 → 不需要图例，标题已说明） ──
    ax = axes[0, 0]
    style_ax(ax)
    ax.plot(losses, color=C_POS, linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("第几轮 (epoch)")
    ax.set_ylabel("损失 L（对数刻度）")
    ax.set_title("训练损失曲线")
    ax.annotate(f"最终 {losses[-1]:.4f}", xy=(len(losses) - 1, losses[-1]),
                xytext=(-95, 18), textcoords="offset points",
                color=C_INK2, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=C_MUTED, linewidth=0.8))

    # ── 图2：预测 vs 真实（测试集） ──
    ax = axes[0, 1]
    style_ax(ax)
    yp_te = predict(X_te, w, b)
    ax.scatter(y_te, yp_te, s=9, alpha=0.35, color=C_POS, edgecolors="none")
    lim = [0, max(y_te.max(), yp_te.max()) * 1.03]
    ax.plot(lim, lim, color=C_MUTED, linewidth=1.2, linestyle="--")
    ax.text(lim[1] * 0.52, lim[1] * 0.44, "完美预测线 y = x",
            color=C_MUTED, fontsize=9, rotation=38)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("真实房价 (10万美元)")
    ax.set_ylabel("预测房价 (10万美元)")
    ax.set_title(f"测试集预测 vs 真实      R² = {r2_score(yp_te, y_te):.4f}")

    # ── 图3：残差图 ──
    ax = axes[1, 0]
    style_ax(ax)
    resid = y_te - yp_te
    ax.scatter(yp_te, resid, s=9, alpha=0.35, color=C_POS, edgecolors="none")
    ax.axhline(0, color=C_MUTED, linewidth=1.2)
    ax.set_xlabel("预测房价 (10万美元)")
    ax.set_ylabel("残差 = 真实 − 预测")
    ax.set_title("残差图      （点应该均匀散布在 0 线上下）")

    # ── 图4：特征权重（有正负 → 发散配色：蓝正 / 红负） ──
    ax = axes[1, 1]
    style_ax(ax)
    idx = np.arange(len(w))
    colors = [C_POS if v >= 0 else C_NEG for v in w]
    ax.bar(idx, w, color=colors, width=0.68)
    ax.axhline(0, color=C_AXIS, linewidth=1.2)
    ax.set_xticks(idx)
    ax.set_xticklabels([FEATURE_CN[n] for n in FEATURE_NAMES],
                       rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("权重 w（特征已标准化，可直接比大小）")
    ax.set_title("每个特征的影响   蓝=推高房价  红=拉低房价")
    for i, v in enumerate(w):
        ax.text(i, v + (0.04 if v >= 0 else -0.09), f"{v:+.2f}",
                ha="center", fontsize=8.5, color=C_INK2)

    fig.suptitle("加州房价 · 线性回归诊断", color=C_INK, fontsize=15, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path = os.path.join(out_dir, "diagnostics.png")
    fig.savefig(path, dpi=115, facecolor=C_SURFACE)
    print(f"\n    四张诊断图已保存: {path}")


def main():
    print("=" * 68)
    print("项目一 · 加州房价预测（线性回归）")
    print("=" * 68)

    # ① 数据
    print("\n[1/5] 加载数据 …")
    df = load_data()
    X_tr, y_tr, X_te, y_te, mean, std = split_and_standardize(df)
    print(f"      训练集 {X_tr.shape}   测试集 {X_te.shape}")

    # ② 训练
    print("\n[2/5] 训练 …")
    w, b, losses = train(X_tr, y_tr, lr=0.1, epochs=2000)
    print(f"      损失 {losses[0]:.4f} → {losses[-1]:.6f}")

    # ③ 评估（两个数据集都要看）
    print("\n[3/5] 评估")
    for name, X, y in [("训练集", X_tr, y_tr), ("测试集", X_te, y_te)]:
        yp = predict(X, w, b)
        print(f"      {name}:  RMSE = {rmse(yp, y):.4f}  ({rmse(yp, y)*10:.1f} 千美元)"
              f"   R² = {r2_score(yp, y):.4f}")

    print("\n      测试集残差分析:")
    yp_te = predict(X_te, w, b)
    residual_report(yp_te, y_te)

    # ④ 预测一个街区的房价
    print("\n[4/5] 对一个新街区做预测")
    demo = np.array([8.3, 41, 6.9, 1.0, 322, 2.5, 37.88, -122.23])
    price = predict_raw(w, b, mean, std, demo)
    print(f"      输入特征（原始量纲）:")
    for n, v in zip(FEATURE_NAMES, demo):
        print(f"        {FEATURE_CN[n]:8s} {v}")
    print(f"\n      预测房价: {price:.4f}  →  约 {price*10:.1f} 万美元")

    # ⑤ 出图
    print("\n[5/5] 画诊断图 …")
    make_figures(w, b, losses, X_tr, y_tr, X_te, y_te)

    print("\n" + "=" * 68)
    print("完成。打开 03_housing/outputs/diagnostics.png 看结果。")
    print("=" * 68)


if __name__ == "__main__":
    main()
