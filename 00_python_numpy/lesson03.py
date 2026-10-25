"""
第 03 课：多变量线性回归 —— 从「一个 w」到「一组 w」
======================================================

第 02 课你写的是：   ŷ = w * x + b        w 是【一个数】
这一课要写的是：     ŷ = X @ w + b        X 是 (N, D) 矩阵，w 是 (D,) 向量

看起来只变了一点点，但这一步跨过去，你就已经站在神经网络门口了。

═══════════════════════════════════════════════════════════════
 为什么要有这一步？
═══════════════════════════════════════════════════════════════

现实里一个东西很少有「一个」特征。比如预测房价，可能同时看：
    面积、房龄、离地铁距离、楼层、朝向 ...   ← D 个特征

每个特征都有自己的权重：
    ŷ = w₁·x₁ + w₂·x₂ + ... + w_D·x_D + b

把它写成矩阵形式就是：
    ŷ = X @ w + b

    X : (N, D)   N 个样本，每个 D 个特征
    w : (D,)     D 个权重
    ŷ : (N,)     N 个预测值

【关键认知】神经网络的一层，做的就是这件事 ——
    只不过它的输出不止一个，所以 w 会变成矩阵 W。

═══════════════════════════════════════════════════════════════
 梯度公式的推导（和 02 课是同一个，只是变向量了）
═══════════════════════════════════════════════════════════════

回忆 02 课标量版的公式：

    d = ŷ - y                    # 误差，形状 (N,)
    dw = (2/N) · Σ dᵢ·xᵢ          # 一个标量
    db = (2/N) · Σ dᵢ

现在 x 变成向量了，所以每个样本的 xᵢ 是一个 D 维向量，
「Σ dᵢ·xᵢ」就变成了「把 N 个 D 维向量按 dᵢ 加权求和」——
这正是矩阵乘法：

    dw = (2/N) · Xᵀ @ d

【形状分析】（写深度学习代码，永远先看形状）
    X   : (N, D)
    X.T : (D, N)
    d   : (N,)
    X.T @ d : (D, N) @ (N,) → (D,)     ✅ 正好和 w 的形状一致！

db 和 02 课一样，还是一个标量：
    db = (2/N) · Σ dᵢ
═══════════════════════════════════════════════════════════════
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK HK",
    "WenQuanYi Zen Hei", "Droid Sans Fallback", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 数据生成（已写好）
# ============================================================
def make_data(n=200, d=3, seed=0):
    """生成多特征数据：y = X @ w_true + b_true + 噪声

    真实答案 w_true = [2.0, -3.0, 0.5]，b_true = 1.5
    程序看不到它们，只能通过 X 和 y 学出来。
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d))                    # (N, D)

    # 前 3 个权重取好记的数，特征更多时往后顺延
    base = np.array([2.0, -3.0, 0.5, 1.7, -1.1, 0.9, -2.3, 1.4])
    w_true = base[:d].copy() if d <= len(base) else rng.normal(0, 2, d)

    b_true = 1.5
    y = X @ w_true + b_true + rng.normal(0, 0.3, n)  # (N,)
    return X, y, w_true, b_true


# ============================================================
# ① 前向：X @ w + b
# ============================================================
def predict(X, w, b):
    """ŷ = X @ w + b

    X : (N, D)
    w : (D,)
    b : 标量
    返回: (N,)

    提示：一行搞定。注意是矩阵乘法 @，不是 *。
    想想为什么：每个样本的预测 = 它 D 个特征分别乘上 D 个权重再求和。
    """
    # TODO ①
    assert X.shape[1] == w.shape[0]
    return X @ w + b
    raise NotImplementedError("还没写 predict")


# ============================================================
# ② 损失：和 02 课一模一样
# ============================================================
def mse_loss(y_pred, y_true):
    """L = mean((y_pred - y_true) ** 2)，返回标量"""
    # TODO
    return np.mean((y_pred - y_true) ** 2)  
    raise NotImplementedError("还没写 mse_loss")


# ============================================================
# ③ 反向：本课的核心 —— 梯度从标量变成向量
# ============================================================
def compute_gradients(X, y_true, y_pred):
    """返回 (dw, db)

        d  = y_pred - y_true        # (N,)   误差
        dw = (2/N) * X.T @ d        # (D,)   ← 注意形状！
        db = (2/N) * np.sum(d)      # 标量

    提示：
      - 先算 d
      - dw 用矩阵乘法：X.T @ d，结果的形状会自动是 (D,)
      - N 用 len(y_true) 或 X.shape[0]
    """
    # TODO ③
    d = y_pred - y_true
    dw = 2 / X.shape[0] * X.T @ d
    db = 2 / X.shape[0] * np.sum(d)
    return dw, db
    raise NotImplementedError("还没写 compute_gradients")


# ============================================================
# ④ 梯度检验（已写好，支持向量）
# ============================================================
def check_gradients(X, y, w, b, eps=1e-6, verbose=True):
    """对 w 的每个分量、以及 b，分别做数值梯度检验。（已写好）"""
    dw_analytic, db_analytic = compute_gradients(X, y, predict(X, w, b))

    loss_at = lambda w_, b_: mse_loss(predict(X, w_, b_), y)

    # w 的每个分量单独扰动一次
    dw_numeric = np.zeros_like(w)
    for i in range(len(w)):
        wp = w.copy(); wp[i] += eps
        wm = w.copy(); wm[i] -= eps
        dw_numeric[i] = (loss_at(wp, b) - loss_at(wm, b)) / (2 * eps)

    db_numeric = (loss_at(w, b + eps) - loss_at(w, b - eps)) / (2 * eps)

    err_w = np.max(np.abs(dw_analytic - dw_numeric) /
                   (np.abs(dw_analytic) + np.abs(dw_numeric) + 1e-12))
    err_b = abs(db_analytic - db_numeric) / (abs(db_analytic) + abs(db_numeric) + 1e-12)

    if verbose:
        print(f"    dw 解析 {np.round(dw_analytic, 6)}")
        print(f"    dw 数值 {np.round(dw_numeric, 6)}")
        print(f"    最大相对误差 dw = {err_w:.2e},  db = {err_b:.2e}")

    assert err_w < 1e-6, f"dw 公式错了！最大相对误差 {err_w:.2e} —— 检查 X.T @ d 的写法"
    assert err_b < 1e-6, f"db 公式错了！相对误差 {err_b:.2e}"


def check_shapes():
    """只检查 predict 的输出形状"""
    X, _, _, _ = make_data(n=20, d=3)
    yp = predict(X, np.ones(3), 0.0)
    assert yp.shape == (20,), f"predict 应返回 (20,)，实际 {yp.shape}"

    X2, _, _, _ = make_data(n=7, d=5)          # 换一组维度再测一次
    yp2 = predict(X2, np.ones(5), 1.0)
    assert yp2.shape == (7,), f"predict 应返回 (7,)，实际 {yp2.shape}"


# ============================================================
# ⑤ 训练循环（和第 02 课一样，照抄即可）
# ============================================================
def train(X, y, lr=0.1, epochs=300, w=None, b=0.0):
    """梯度下降训练，返回 (w, b, losses)

    和第 02 课的四步循环完全相同，唯一区别：
      - w 现在是向量，初始化用 np.zeros(D)
      - 别的没变
    """
    # TODO ⑤
    if w is None:
        w = np.zeros((X.shape[1]))
    losses = []
    for epoch in range(epochs):
        y_pred = predict(X, w, b)
        loss = mse_loss(y_pred, y)
        losses.append(loss)
        dw, db = compute_gradients(X, y, y_pred)
        w = w - lr * dw
        b = b - lr * db
    return (w, b, losses)
    raise NotImplementedError("还没写 train")


# ============================================================
# 画图（已写好）
# ============================================================
def plot_results(w, b, w_true, b_true, losses,
                 path="00_python_numpy/lesson03_result.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    idx = np.arange(len(w))
    width = 0.35
    axes[0].bar(idx - width/2, w_true, width, label="真实 w", alpha=0.85)
    axes[0].bar(idx + width/2, w, width, label="学到的 w", alpha=0.85)
    axes[0].set_xticks(idx)
    axes[0].set_xticklabels([f"w{i}" for i in idx])
    axes[0].axhline(0, color="k", linewidth=0.8)
    axes[0].legend()
    axes[0].set_title(f"每个特征的权重    (b: 学到 {b:.3f} / 真实 {b_true})")
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].plot(losses, linewidth=2)
    axes[1].set_xlabel("第几轮 (epoch)")
    axes[1].set_ylabel("损失 L")
    axes[1].set_title("损失曲线")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=110)
    print(f"    图片已保存: {path}")


# ============================================================
# 检查（不用改）
# ============================================================
def check_loss():
    got = mse_loss(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    assert abs(got - 4.0 / 3.0) < 1e-12, f"应为 4/3，实际 {got}"


def check_grad():
    X, y, _, _ = make_data(n=50, d=4)

    # 先检查梯度【形状】—— 这一步最容易错（忘了转置就会得到 (N,) 而不是 (D,)）
    dw, db = compute_gradients(X, y, predict(X, np.ones(4), 0.0))
    assert dw.shape == (4,), \
        f"dw 形状应为 (4,)（和 w 一致），实际 {dw.shape} —— X.T @ d 里的转置写了吗？"
    assert np.ndim(db) == 0, f"db 应是标量，实际形状 {np.shape(db)}"

    # 再检查梯度【数值】
    check_gradients(X, y, w=np.array([1.2, -0.7, 2.5, 0.3]), b=-0.6)
    check_gradients(X, y, w=np.array([-1.0, 0.4, -2.2, 1.8]), b=2.1)


def check_train():
    X, y, w_true, b_true = make_data(n=200, d=3)
    w, b, losses = train(X, y, lr=0.1, epochs=300)

    assert len(losses) == 300, f"losses 应有 300 个值，实际 {len(losses)}"
    assert losses[-1] < losses[0], "损失没有下降"
    assert losses[-1] < 0.2, f"最终损失 {losses[-1]:.4f} 太大，应 < 0.2（噪声水平约 0.3²=0.09）"

    print(f"    学到:  w = {np.round(w, 4)},  b = {b:.4f}")
    print(f"    真实:  w = {w_true},  b = {b_true}")
    err = np.max(np.abs(w - w_true))
    assert err < 0.15, f"w 误差最大 {err:.4f}，应 < 0.15"

    return w, b, losses


def main():
    checks = [
        ("① 前向 predict（含形状检查）", check_shapes),
        ("② 损失 mse_loss", check_loss),
        ("③ 梯度公式（向量版，含梯度检验）", check_grad),
        ("⑤ 训练循环 train", check_train),
    ]

    passed = 0
    result = None
    for name, fn in checks:
        try:
            r = fn()
            if name.startswith("⑤"):
                result = r
            print(f"[PASS] {name}")
            passed += 1
        except NotImplementedError as e:
            print(f"[SKIP] {name} —— {e}")
        except AssertionError as e:
            print(f"[FAIL] {name} —— {e}")
        except Exception as e:
            tb = e.__traceback__
            while tb.tb_next:
                tb = tb.tb_next
            print(f"[ERROR] {name} —— 第 {tb.tb_lineno} 行出错: {type(e).__name__}: {e}")

    print(f"\n通过 {passed} / {len(checks)}")

    if passed == len(checks):
        w, b, losses = result
        _, _, w_true, b_true = make_data()
        plot_results(w, b, w_true, b_true, losses)
        print("\n全部通过！打开 00_python_numpy/lesson03_result.png")
        print("左边应该看到：学到的 w 和真实的 w 几乎一样高。")


if __name__ == "__main__":
    main()
