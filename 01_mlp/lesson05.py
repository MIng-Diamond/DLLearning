"""
第 05 课：手写 MLP —— 第一次让网络「弯」起来
================================================

【动机】
你之前写的模型是  ŷ = X @ w + b —— 它只能拟合直线（或平面）。
无论怎么调 w 和 b，都拟合不出 y = sin(x) 这样的曲线。

【解法】加一层「隐藏层 + 激活函数」：

    x (N,1)
      │
      ├─ z1 = x @ W1 + b1        线性变换        (N, H)
      ├─ a1 = relu(z1)           非线性激活 ← 关键！ (N, H)
      ├─ y_pred = a1 @ W2 + b2   线性变换        (N, 1)
      │
    L = mean((y_pred - y)²)

【为什么必须要有 relu？】
    如果没有它，两层线性叠起来还是线性：
        W2 @ (W1 @ x) = (W2 @ W1) @ x     ← 又变回一个矩阵，白搭
    加了非线性，网络才有能力拟合弯曲的函数。

【ReLU】
    relu(z) = max(0, z)          负数变 0，正数不变
    导数：  z > 0 → 1 ；  z ≤ 0 → 0

═══════════════════════════════════════════════════════════════
 反向传播推导（本课核心）
═══════════════════════════════════════════════════════════════

前向：
    z1 = x @ W1 + b1        (N, H)
    a1 = relu(z1)           (N, H)
    yp = a1 @ W2 + b2       (N, 1)
    L  = mean((yp - y)²)    (标量)

反向（从 L 往回走，每一步都做「链式法则 × 局部导数」）：

    ① dyp = 2 * (yp - y) / N           (N, 1)   L 对 yp 的梯度（MSE 的导数）

    ② dW2 = a1.T @ dyp                 (H, 1)   ┐ yp = a1 @ W2 + b2，
       db2 = np.sum(dyp, axis=0)       (1,)     ┘ 所以对 W2 的梯度 = a1.T @ dyp

    ③ da1 = dyp @ W2.T                 (N, H)   误差往回传到 a1

    ④ dz1 = da1 * (z1 > 0)             (N, H)   ← 穿过 relu！
                                                (z1 > 0) 就是 relu 的导数：正数为 1，负数为 0

    ⑤ dW1 = x.T @ dz1                  (1, H)   ┐ z1 = x @ W1 + b1，
       db1 = sum(dz1, axis=0)          (H,)     ┘ 和 ② 是同一个套路

【形状分析】每个梯度的形状都必须和对应的参数【完全一致】：
    dW1 (1,H) ↔ W1 (1,H)      db1 (H,) ↔ b1 (H,)
    dW2 (H,1) ↔ W2 (H,1)      db2 (1,) ↔ b2 (1,)

【db 的规律】np.sum(dz, axis=0) —— 沿 batch 维（axis 0）求和，
            剩下的就是「每个神经元一个偏置梯度」。
═══════════════════════════════════════════════════════════════
"""

import os
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
# 数据 & 参数初始化
# ============================================================
def make_data(n=200, seed=0):
    """生成 y = sin(x) 的数据。x ∈ [-3, 3]，形状都是 (N, 1)"""
    rng = np.random.default_rng(seed)
    x = np.linspace(-3, 3, n).reshape(-1, 1)      # (N, 1)
    y = np.sin(x) + rng.normal(0, 0.05, (n, 1))   # (N, 1) 加一点噪声
    return x, y


def init_params(h=32, seed=0):
    """随机初始化参数。

    ⚠️ 千万不能全零初始化！如果 W1 全是 0，所有隐藏单元算出来的东西
       完全一样，它们会永远保持一样，等于只有一个单元在工作。
       随机初始化才能让每个单元学到不同的东西。

    我们用一个合适的缩放：W1 用 N(0,1)，W2 用 N(0, 1/sqrt(h))
    """
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1.0, (1, h))                # (1, H)
    b1 = np.zeros(h)                               # (H,)
    W2 = rng.normal(0, 1.0 / np.sqrt(h), (h, 1))   # (H, 1)
    b2 = np.zeros(1)                               # (1,)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


# ============================================================
# ① 激活函数
# ============================================================
def relu(z):
    """relu(z) = max(0, z)

    提示：np.maximum(0, z) 可以逐元素取最大值。
    （注意不是 np.max —— 那个会把整个数组压成一个数）
    """
    # TODO ①
    return np.maximum(0, z)
    raise NotImplementedError("还没写 relu")


# ============================================================
# ② 前向
# ============================================================
def forward(x, p):
    """返回 (z1, a1, y_pred)

        z1 = x @ W1 + b1        (N, H)   线性变换（未激活）
        a1 = relu(z1)           (N, H)   激活
        yp = a1 @ W2 + b2       (N, 1)   输出

    注意 z1 也要返回！因为反向传播穿过 relu 时需要用到它
    （靠 z1 > 0 判断导数）。
    """
    # TODO ②（三行）
    z1 = x @ p["W1"] + p["b1"]
    a1 = relu(z1)
    yp = a1 @ p["W2"] + p["b2"]
    return (z1, a1, yp)
    raise NotImplementedError("还没写 forward")


# ============================================================
# ③ 反向传播（本课核心）
# ============================================================
def compute_gradients(x, y, p, cache):
    """返回一个字典，装着 dW1, db1, dW2, db2。

    cache 是 forward 返回的 (z1, a1, y_pred)。

    照着上面【反向传播推导】那五步写。每一步都先想清楚形状：
        dyp  : (N, 1)
        dW2  : (H, 1)   = a1.T @ dyp
        db2  : (1,)     = np.sum(dyp, axis=0)
        da1  : (N, H)   = dyp @ W2.T
        dz1  : (N, H)   = da1 * (z1 > 0)      ← 穿过 relu
        dW1  : (1, H)   = x.T @ dz1
        db1  : (H,)     = np.sum(dz1, axis=0)

    提示：
      - np.sum(dz1, axis=0) 是沿 batch 维求和，结果是每个隐藏单元一个梯度
      - (z1 > 0) 得到的是布尔数组，和 da1 相乘时会自动变成 0/1
    """
    z1, a1, y_pred = cache
    N = len(y)

    # TODO ③
    z1, a1, y_pred = cache
    dyp = 2 / y.shape[0] * (y_pred - y)
    dW2 = a1.T @ dyp
    db2 = np.sum(dyp, axis=0)
    da1 = dyp @ p["W2"].T
    dz1 = da1 * (z1 > 0)
    dW1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0)
    return {"dW2": dW2, "db2": db2, "dW1": dW1, "db1": db1}
    raise NotImplementedError("还没写 compute_gradients")


def mse_loss(y_pred, y_true):
    """L = mean((y_pred - y_true) ** 2)  —— 和第 02/03 课一样"""
    return np.mean((y_pred - y_true) ** 2)


# ============================================================
# ④ 梯度检验（已写好，支持字典形式的参数）
# ============================================================
def check_gradients(x, y, p, eps=1e-6, verbose=True):
    """数值梯度 vs 解析梯度，逐个参数检查。（已写好）"""
    z1, a1, yp = forward(x, p)
    grads = compute_gradients(x, y, p, (z1, a1, yp))

    loss_at = lambda pp: mse_loss(forward(x, pp)[2], y)
    worst = 0.0

    for name in ["W1", "b1", "W2", "b2"]:
        param = p[name]
        g_analytic = grads["d" + name]
        assert g_analytic.shape == param.shape, \
            f"d{name} 形状应为 {param.shape}（和 {name} 一致），实际 {g_analytic.shape}"

        g_numeric = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            old = param[idx]
            param[idx] = old + eps; lp = loss_at(p)
            param[idx] = old - eps; lm = loss_at(p)
            param[idx] = old                                   # 还原
            g_numeric[idx] = (lp - lm) / (2 * eps)
            it.iternext()

        denom = np.abs(g_analytic) + np.abs(g_numeric) + 1e-12
        err = np.max(np.abs(g_analytic - g_numeric) / denom)
        worst = max(worst, err)
        if verbose:
            print(f"    d{name:3s}  最大相对误差 = {err:.2e}")

    assert worst < 1e-6, f"梯度公式有错！最大相对误差 {worst:.2e}，应 < 1e-6"


# ============================================================
# ⑤ 训练循环（和第 02/03 课完全一样，只是参数变多了）
# ============================================================
def train(x, y, h=32, lr=0.05, epochs=2000, seed=0):
    """返回 (params, losses)

    四步循环完全没变：
        ① y_pred = forward(x, p)[2]
        ② loss   = mse_loss(y_pred, y)
        ③ grads  = compute_gradients(x, y, p, cache)
        ④ 用 grads 更新 p 里的每个参数

    ⚠️ 注意 ③ 要拿到 forward 的中间结果，所以这一课把 forward 的返回值
       存成 cache 传进去。
    """
    # TODO ⑤
    losses = []
    p = init_params(h, seed)
    for epoch in range(epochs):

        cache = forward(x, p)
        y_pred = cache[2]
        loss = mse_loss(y_pred, y)
        losses.append(loss)
        grad = compute_gradients(x, y, p, cache)
        p["W1"] = p["W1"] - lr * grad["dW1"]
        p["W2"] = p["W2"] - lr * grad["dW2"]
        p["b1"] = p["b1"] - lr * grad["db1"]
        p["b2"] = p["b2"] - lr * grad["db2"]
    return (p, losses)   
    raise NotImplementedError("还没写 train")


# ============================================================
# 画图（已写好）
# ============================================================
def plot_results(x, y, y_pred, losses, path=None):
    # 图片存到本文件旁边，这样不管从哪个目录运行都不会找不到路径
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "lesson05_result.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].scatter(x, y, s=10, alpha=0.4, label="数据 (y = sin(x) + 噪声)")
    order = np.argsort(x.ravel())
    axes[0].plot(x.ravel()[order], y_pred.ravel()[order], "r-",
                 linewidth=2.5, label="网络学到的曲线")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].legend(); axes[0].set_title("MLP 拟合结果")
    axes[0].grid(alpha=0.3)

    axes[1].plot(losses, linewidth=2)
    axes[1].set_xlabel("第几轮 (epoch)"); axes[1].set_ylabel("损失 L")
    axes[1].set_title("损失曲线"); axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=110)
    print(f"    图片已保存: {path}")


# ============================================================
# 检查
# ============================================================
def check_relu():
    z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    got = relu(z)
    assert got.shape == z.shape, f"形状应保持 {z.shape}，实际 {got.shape}"
    assert np.allclose(got, [0, 0, 0, 0.5, 2.0]), f"relu 结果不对：{got}"


def check_forward():
    x, y = make_data(n=10)
    p = init_params(h=8)
    z1, a1, yp = forward(x, p)
    assert z1.shape == (10, 8), f"z1 应为 (10,8)，实际 {z1.shape}"
    assert a1.shape == (10, 8), f"a1 应为 (10,8)，实际 {a1.shape}"
    assert yp.shape == (10, 1), f"y_pred 应为 (10,1)，实际 {yp.shape}"
    assert np.all(a1 >= 0), "relu 之后不应有负数"


def check_grad():
    x, y = make_data(n=40)
    p = init_params(h=6, seed=1)
    check_gradients(x, y, p)


def check_train():
    x, y = make_data(n=200)
    p, losses = train(x, y, h=32, lr=0.05, epochs=2000)

    assert len(losses) == 2000, f"losses 应有 2000 个，实际 {len(losses)}"
    assert losses[-1] < losses[0], "损失没有下降"
    print(f"    损失:  {losses[0]:.5f}  →  {losses[-1]:.6f}")
    assert losses[-1] < 0.01, f"最终损失 {losses[-1]:.6f} 太大，应 < 0.01"

    # 对比：一个纯线性模型能做到多好？
    xf = np.hstack([x, np.ones((len(x), 1))])
    w_lin = np.linalg.lstsq(xf, y, rcond=None)[0]
    lin_loss = mse_loss(xf @ w_lin, y)
    print(f"    对比：最优【线性】模型的损失 = {lin_loss:.5f}")
    print(f"          MLP 的损失            = {losses[-1]:.6f}   ← 好得多")
    assert losses[-1] < lin_loss / 5, "MLP 应该明显好于线性模型"

    return p, losses


def main():
    checks = [
        ("① 激活函数 relu", check_relu),
        ("② 前向 forward", check_forward),
        ("③ 反向传播（含梯度检验）", check_grad),
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
        p, losses = result
        x, y = make_data(n=200)
        yp = forward(x, p)[2]
        plot_results(x, y, yp, losses)
        print("\n全部通过！打开 01_mlp/lesson05_result.png")
        print("红色曲线应该漂亮地穿过那些蓝色散点 —— 这是你的第一个神经网络。")


if __name__ == "__main__":
    main()
