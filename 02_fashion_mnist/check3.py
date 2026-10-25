"""
项目 · 第 3 节 —— 检查
========================

跑这个文件，验证 eval_model.py 的三个 TODO。

    cd ~/projects/learn/DeepLearning/02_fashion_mnist
    python check3.py

⚠️ 需要先有 train.py 生成的 mlp.pt。没有的话先跑：
    python train.py
"""

import os
import numpy as np
import torch

from data import get_loaders, CLASSES
from eval_model import (
    load_trained_model,
    collect_predictions,
    confusion_matrix,
    per_class_accuracy,
)


def check_eval():
    assert os.path.exists("mlp.pt"), \
        "找不到 mlp.pt，请先在 02_fashion_mnist 目录下运行 python train.py"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trained_model("mlp.pt", device)
    _, test_loader = get_loaders(batch_size=256)

    # ---------- ① 收集预测 ----------
    preds, labels = collect_predictions(model, test_loader, device)
    assert isinstance(preds, np.ndarray), f"preds 应是 numpy 数组，实际 {type(preds)}"
    assert isinstance(labels, np.ndarray), f"labels 应是 numpy 数组，实际 {type(labels)}"
    assert preds.shape == (10000,), f"preds 形状应为 (10000,)，实际 {preds.shape}"
    assert labels.shape == (10000,), f"labels 形状应为 (10000,)，实际 {labels.shape}"
    assert preds.min() >= 0 and preds.max() <= 9, "preds 的取值应在 0~9"
    assert labels.min() >= 0 and labels.max() <= 9, "labels 的取值应在 0~9"

    # ---------- ② 混淆矩阵 ----------
    cm = confusion_matrix(preds, labels)
    assert cm.shape == (10, 10), f"混淆矩阵形状应为 (10,10)，实际 {cm.shape}"
    assert cm.sum() == 10000, f"混淆矩阵总和应为 10000，实际 {cm.sum()}"
    assert np.array_equal(cm.sum(axis=1), np.bincount(labels, minlength=10)), \
        "混淆矩阵每行之和应等于该类真实样本数"

    overall = cm.trace() / cm.sum()
    direct = (preds == labels).mean()
    assert abs(overall - direct) < 1e-9, \
        f"两种算法不一致：混淆矩阵 {overall:.6f} vs 直接比较 {direct:.6f}"
    assert overall > 0.80, f"总体准确率只有 {overall:.3f}，应 > 0.80"

    # ---------- ③ 每类准确率 ----------
    acc = per_class_accuracy(cm)
    assert acc.shape == (10,), f"每类准确率长度应为 10，实际 {acc.shape}"
    assert np.all(acc >= 0) and np.all(acc <= 1), "每类准确率应在 0~1"
    expected = cm.diagonal() / cm.sum(axis=1)
    assert np.allclose(acc, expected), "每类准确率的公式不对"

    # ---------- 打印结果 ----------
    print(f"    总体准确率: {overall:.4f}")
    worst = np.argsort(acc)[:3]
    print("    最差的 3 类:")
    for i in worst:
        print(f"      {CLASSES[i]:12s} {acc[i]:.4f}")

    return overall, acc


def main():
    try:
        check_eval()
        print("[PASS] 评估与可视化")
        print("\n第 3 节完成 ✅ 可以打开三张图看看模型错在哪")
    except NotImplementedError as e:
        print(f"[SKIP] 评估与可视化 —— {e}")
    except AssertionError as e:
        print(f"[FAIL] 评估与可视化 —— {e}")
    except Exception as e:
        tb = e.__traceback__
        while tb.tb_next:
            tb = tb.tb_next
        print(f"[ERROR] 评估与可视化 —— 第 {tb.tb_lineno} 行: "
              f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
