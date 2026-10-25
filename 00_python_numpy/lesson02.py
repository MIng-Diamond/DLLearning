"""
第 02 课：手写线性回归 + 梯度下降
=====================================

这是你人生第一个「会自己学习的程序」。

任务：给一堆带噪声的散点，让程序自己找出背后的直线 y = 3x + 2。
      程序一开始不知道 3 和 2，它从 w=0, b=0 开始，靠梯度下降一步步逼近。

═══════════════════════════════════════════════════════════════
 数学推导（先看懂这个，再写代码）
═══════════════════════════════════════════════════════════════

【模型】    ŷᵢ = w · xᵢ + b          对第 i 个样本的预测

【损失】    L = (1/N) · Σ (ŷᵢ - yᵢ)²   均方误差 MSE

【梯度】    我们要求 L 对 w 和 b 的偏导数。

  设 dᵢ = ŷᵢ - yᵢ  （第 i 个样本的预测误差）
  那么 L = (1/N) · Σ dᵢ²

  ▸ 对 w 求导：
      dL/dw = (1/N) · Σ 2·dᵢ · (ddᵢ/dw)
    因为 dᵢ = w·xᵢ + b - yᵢ，所以 ddᵢ/dw = xᵢ
      dL/dw = (2/N) · Σ dᵢ · xᵢ          ← Σ 里每项是「误差 × 对应的 x」

  ▸ 对 b 求导：
     ddᵢ/db = 1
      dL/db = (2/N) · Σ dᵢ               ← Σ 里每项就是「误差」

【更新】    w ← w - lr · dL/dw
            b ← b - lr · dL/db
            lr 是学习率（learning rate），控制每步走多大

═══════════════════════════════════════════════════════════════
 你的任务：把下面所有  # TODO  填完
═══════════════════════════════════════════════════════════════
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 不弹窗，直接存文件
import matplotlib.pyplot as plt

# 让 matplotlib 能显示中文（否则图上的中文会变成方块 □□□）
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK HK",
    "WenQuanYi Zen Hei", "Droid Sans Fallback", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False   # 让负号正常显示


# ============================================================
# 数据生成（已写好，不用改）
# ============================================================
def make_data(n=100, w_true=3.0, b_true=2.0, noise=0.5, seed=0):
    """生成 y = w_true * x + b_true + 噪声。

    真实答案藏在 w_true=3.0, b_true=2.0 里 —— 但你的程序【看不到】这两个数，
    它只能看到 x 和 y，要靠学习把它们猜出来。
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n)                       # x 均匀分布在 [-3, 3]
    y = w_true * x + b_true + rng.normal(0, noise, n)   # 加高斯噪声
    return x, y


# ============================================================
# ① 前向：用当前参数算预测值
# ============================================================
def predict(x, w, b):
    """ŷ = w * x + b

    x : shape (N,)   N 个样本
    w : 标量
    b : 标量
    返回: shape (N,)  N 个预测值

    提示：这一行代码就够了，NumPy 会自动对每个 x 元素算一遍。
    """
    # TODO ①
    return w * x + b
    raise NotImplementedError("还没写 predict")


# ============================================================
# ② 损失：预测得有多差
# ============================================================
def mse_loss(y_pred, y_true):
    """L = mean((y_pred - y_true) ** 2)

    返回一个标量（不是数组）。

    提示：先算差值数组，平方，再取平均。用 np.mean。
    """
    # TODO ②
    return np.mean((y_pred - y_true) ** 2)
    raise NotImplementedError("还没写 mse_loss")


# ============================================================
# ③ 反向：算梯度（本课的核心）
# ============================================================
def compute_gradients(x, y_true, y_pred):
    """返回 (dw, db)，即 dL/dw 和 dL/db。

    照着上面推导的公式写：

        dᵢ = y_pred - y_true          # 误差，shape (N,)
        dw = (2/N) * Σ dᵢ · xᵢ
        db = (2/N) * Σ dᵢ

    N 是样本个数。

    提示：
      - 误差数组可以直接整体运算，不用写循环
      - 求和用 np.sum
      - 两个都是标量
    """
    # TODO ③
    di = y_pred - y_true
    tmp1 = di * x
    tmp2 = di.sum()
    dw = 2 / y_true.shape[0] * tmp1.sum()
    db = 2 / y_true.shape[0] * tmp2
    return (dw, db)
    raise NotImplementedError("还没写 compute_gradients")


# ============================================================
# ④ 梯度检验：用数值方法验证你的公式推对了没有
# ============================================================
def numerical_gradient(f, param, eps=1e-6):
    """数值梯度：不求导，用「微小扰动」近似导数。

        df/dp ≈ ( f(p + eps) - f(p - eps) ) / (2 * eps)

    f     : 一个只接受【一个标量】的函数，返回一个标量
    param : 要求导的那个点
    eps   : 扰动大小

    这是整个深度学习里最重要的调试工具。
    如果手推的梯度算错了，它一定能抓出来。
    """
    # TODO ④
    return (f(param + eps) - f(param - eps)) / (2 * eps)
    raise NotImplementedError("还没写 numerical_gradient")


def check_gradients(x, y, w, b, verbose=True):
    """把上面的解析梯度和数值梯度对比。（已写好，不用改）

    原理：如果你推的公式是对的，两个结果应该几乎完全相等。
    """
    y_pred = predict(x, w, b)
    dw_analytic, db_analytic = compute_gradients(x, y, y_pred)

    # 把 w 和 b 分别"冻住另一个、只动自己"，做成单变量函数
    loss_w = lambda w_: mse_loss(predict(x, w_, b), y)
    loss_b = lambda b_: mse_loss(predict(x, w, b_), y)

    dw_numeric = numerical_gradient(loss_w, w)
    db_numeric = numerical_gradient(loss_b, b)

    # 相对误差：差值除以两边量级，避免受数值大小影响
    err_w = abs(dw_analytic - dw_numeric) / (abs(dw_analytic) + abs(dw_numeric) + 1e-12)
    err_b = abs(db_analytic - db_numeric) / (abs(db_analytic) + abs(db_numeric) + 1e-12)

    if verbose:
        print(f"    dL/dw  解析 {dw_analytic:+.8f} | 数值 {dw_numeric:+.8f} | 相对误差 {err_w:.2e}")
        print(f"    dL/db  解析 {db_analytic:+.8f} | 数值 {db_numeric:+.8f} | 相对误差 {err_b:.2e}")

    assert err_w < 1e-6, f"dL/dw 公式错了！相对误差 {err_w:.2e}（应该 < 1e-6）——回去检查公式"
    assert err_b < 1e-6, f"dL/db 公式错了！相对误差 {err_b:.2e}（应该 < 1e-6）——回去检查公式"


# ============================================================
# ⑤ 训练循环：四步循环重复 N 次
# ============================================================
def train(x, y, lr=0.1, epochs=200, w=0.0, b=0.0):
    """梯度下降训练。返回 (w, b, losses)。

    losses 是一个列表，记录每一轮的损失值，最后用来画曲线。

    每一轮做四件事：
        ① y_pred = predict(x, w, b)
        ② loss   = mse_loss(y_pred, y)
        ③ dw, db = compute_gradients(x, y, y_pred)
        ④ 用 dw, db 更新 w 和 b

    注意：更新时两个参数要【同时】更新 —— 也就是先算完两个新值再赋值，
          不能先更新 w 再用新的 w 去算 db（本函数的写法天然没这个问题，
          因为 dw/db 在第③步已经一起算出来了）。

    提示：把 loss 存进 losses 列表；更新用  w = w - lr * dw
    """
    # TODO ⑤
    losses = []
    for epoch in range(epochs):
        y_pred = predict(x, w, b)
        loss = mse_loss(y_pred, y)
        losses.append(loss)
        dw, db = compute_gradients(x, y, y_pred)
        w = w - lr * dw
        b = b - lr * db

    return (w, b, losses)
    raise NotImplementedError("还没写 train")


# ============================================================
# 画图（已写好，不用改）
# ============================================================
def plot_results(x, y, w, b, losses, path="00_python_numpy/lesson02_result.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].scatter(x, y, s=18, alpha=0.6, label="数据点")
    xs = np.linspace(x.min(), x.max(), 100)
    axes[0].plot(xs, w * xs + b, "r-", linewidth=2.5, label=f"学到的: y = {w:.3f}x + {b:.3f}")
    axes[0].plot(xs, 3.0 * xs + 2.0, "g--", linewidth=2, label="真实: y = 3.000x + 2.000")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()
    axes[0].set_title("拟合结果")
    axes[0].grid(alpha=0.3)

    axes[1].plot(losses, linewidth=2)
    axes[1].set_xlabel("第几轮 (epoch)")
    axes[1].set_ylabel("损失 L")
    axes[1].set_title("损失曲线（应该一直往下掉）")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=110)
    print(f"    图片已保存: {path}")


# ============================================================
# 检查（不用改）
# ============================================================
def check_predict():
    x = np.array([0.0, 1.0, 2.0])
    got = predict(x, 3.0, 2.0)
    assert isinstance(got, np.ndarray), f"应该返回 numpy 数组，实际是 {type(got)}"
    assert got.shape == (3,), f"形状应为 (3,)，实际 {got.shape}"
    assert np.allclose(got, [2.0, 5.0, 8.0]), f"预测值不对：{got}（w=3,b=2,x=[0,1,2] 应为 [2,5,8]）"


def check_loss():
    y_pred = np.array([1.0, 2.0, 3.0])
    y_true = np.array([1.0, 2.0, 5.0])
    got = mse_loss(y_pred, y_true)
    assert np.isscalar(got) or np.ndim(got) == 0, f"应该返回标量，实际 {got}"
    assert abs(got - 4.0 / 3.0) < 1e-12, f"应为 4/3 ≈ 1.3333，实际 {got}"


def check_gradients_formula():
    x, y = make_data(n=50)
    check_gradients(x, y, w=1.7, b=-0.4)
    check_gradients(x, y, w=-2.3, b=1.1)


def check_train():
    x, y = make_data(n=100)
    w, b, losses = train(x, y, lr=0.1, epochs=200)

    assert len(losses) == 200, f"losses 应有 200 个值，实际 {len(losses)}"
    assert losses[-1] < losses[0], "损失没有下降，训练没生效"
    assert losses[-1] < 0.5, f"最终损失 {losses[-1]:.4f} 太大，应该 < 0.5"

    # 检查是否单调下降（允许极小的数值波动）
    bad = [i for i in range(1, len(losses)) if losses[i] > losses[i - 1] * 1.001]
    assert not bad, f"损失不单调下降，在第 {bad[:5]} 轮出现上升 —— 学习率太大了？"

    print(f"    学到:  w = {w:.4f}, b = {b:.4f}")
    print(f"    真实:  w = 3.0000, b = 2.0000")
    print(f"    损失:  {losses[0]:.4f}  →  {losses[-1]:.6f}")
    assert abs(w - 3.0) < 0.2, f"学到的 w={w:.4f} 离 3.0 太远（误差应 < 0.2）"
    assert abs(b - 2.0) < 0.2, f"学到的 b={b:.4f} 离 2.0 太远（误差应 < 0.2）"


def main():
    checks = [
        ("① 前向 predict", check_predict),
        ("② 损失 mse_loss", check_loss),
        ("③ 梯度公式（含梯度检验）", check_gradients_formula),
        ("⑤ 训练循环 train", check_train),
    ]

    passed = 0
    for name, fn in checks:
        try:
            fn()
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
        # 全部通过 → 画图看结果
        x, y = make_data(n=100)
        w, b, losses = train(x, y, lr=0.1, epochs=200)
        plot_results(x, y, w, b, losses)
        print("\n全部通过！打开 00_python_numpy/lesson02_result.png 看看你的成果。")


if __name__ == "__main__":
    main()
