"""
阶段 0 · 通关测试（白纸测试）
================================================================

【规则】
  1. 不许打开 lesson01 / lesson02 / lesson03 / lesson04
  2. 不许上网查
  3. 可以查 numpy 的【API 用法】（比如 help(np.mean) 看参数怎么传），
     但不要查算法本身的推导
  4. 凭记忆写。写不出来就空着，不要硬凑 —— 空着才有信息量

【为什么要做这个】
  能「读懂」和能「写出来」是两种完全不同的能力。
  前面的课你都在填空，现在要自己从零搭一遍。

  卡住是好事 —— 那正是这个测试的价值：它告诉你哪里还没真正掌握。

【限时】建议 60 分钟。不强制，但记一下你花了多久，写在下面的横线上。

  我花了 ______ 分钟

【合格标准】
  5 个函数全部写出来，且底部的检查全过。

【怎么跑】
  python 00_python_numpy/checkpoint_stage0.py
"""

import sys
import numpy as np


# ============================================================
# 1. softmax —— 数值稳定版
# ============================================================
def softmax(x, axis=-1):
    """把一组数变成概率分布（每一组加起来 = 1）。

    要求：数值稳定。（即：输入里有 1000 这种大数时，不能返回 nan）

    参数 x    : 任意形状的数组
    参数 axis : 沿着哪一维做归一化
    返回      : 和 x 同形状
    """
    # 你的代码
    x_max = x.max(axis, keepdims = True)
    e = np.exp(x - x_max)
    return e / e.sum(axis, keepdims = True)


# ============================================================
# 2. 前向
# ============================================================
def predict(X, w, b):
    """线性预测。

    X : (N, D)      N 个样本，D 个特征
    w : (D,)        D 个权重
    b : 标量        偏置
    返回: (N,)      每个样本一个预测值
    """
    # 你的代码
    assert X.shape[1] == w.shape[0]
    return X @ w + b

# ============================================================
# 3. 损失
# ============================================================
def mse_loss(y_pred, y_true):
    """均方误差。返回一个标量。"""
    # 你的代码
    return np.mean((y_pred - y_true) ** 2)

# ============================================================
# 4. 梯度（手推，不许用 autograd）
# ============================================================
def compute_gradients(X, y_true, y_pred):
    """返回 (dw, db)，即 dL/dw 和 dL/db。

    X       : (N, D)
    y_true  : (N,)
    y_pred  : (N,)
    dw      : (D,)     必须和 w 同形状
    db      : 标量

    提示：先想清楚形状，再写代码。
    """
    # 你的代码
    L = y_pred - y_true
    dw = 2 / X.shape[0] * X.T @ L
    db = 2 / X.shape[0] * L.sum() 
    return dw, db
# ============================================================
# 5. 训练循环
# ============================================================
def train(X, y, lr=0.1, epochs=300, w=None, b=0.0):
    """梯度下降训练。返回 (w, b, losses)。

    losses: 列表，记录每一轮的损失

    w=None 表示调用者没传，你需要自己初始化（想想该初始化成什么形状）
    """
    # 你的代码

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
    return w, b, losses
# ============================================================
#   以 下 是 检 查 部 分 ， 不 要 修 改
# ============================================================
def _numerical_gradient(f, param, eps=1e-6):
    return (f(param + eps) - f(param - eps)) / (2 * eps)


def _make_data(n=200, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d))
    # 前 3 个权重好记，特征更多时往后顺延
    base = np.array([2.0, -3.0, 0.5, 1.7, -1.1, 0.9, -2.3, 1.4])
    w_true = base[:d].copy() if d <= len(base) else rng.normal(0, 2, d)
    y = X @ w_true + 1.5 + rng.normal(0, 0.3, n)
    return X, y, w_true


def _check_softmax():
    rng = np.random.default_rng(0)

    x = rng.standard_normal((4, 5))
    s = softmax(x, axis=1)
    assert s.shape == (4, 5), f"形状不对：{s.shape}"
    assert np.allclose(s.sum(axis=1), 1.0), f"每行和应为 1，实际 {s.sum(axis=1)}"
    assert np.all(s >= 0), "概率不能为负"

    big = np.array([[1000.0, 1001.0, 1002.0]])
    sb = softmax(big, axis=1)
    assert np.all(np.isfinite(sb)), f"大数输入产生了 nan/inf：{sb}（没做数值稳定处理）"
    assert np.allclose(sb.sum(axis=1), 1.0), "大数输入时每行和应为 1"

    x2 = rng.standard_normal((3, 4))
    assert np.allclose(softmax(x2, axis=0).sum(axis=0), 1.0), "axis=0 时不对"

    manual = np.exp(big - 1002.0) / np.exp(big - 1002.0).sum()
    assert np.allclose(sb[0], manual), "数值和手算不一致"


def _check_predict():
    # 用单位矩阵当 X，这样每个样本只挑出一个权重，结果一眼能算出来
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    w = np.array([3.0, -2.0])
    got = predict(X, w, 1.0)
    assert got.shape == (2,), f"形状应为 (2,)，实际 {got.shape}"
    assert np.allclose(got, [4.0, -1.0]), f"数值不对：{got}，应为 [4, -1]（3+1 和 -2+1）"

    X2, _, _ = _make_data(n=7, d=5)
    assert predict(X2, np.ones(5), 0.0).shape == (7,), "换维度后形状不对"


def _check_loss():
    got = mse_loss(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    assert np.ndim(got) == 0, f"应返回标量，实际 {got}"
    assert abs(got - 4.0 / 3.0) < 1e-12, f"应为 4/3，实际 {got}"


def _check_gradients():
    X, y, _ = _make_data(n=50, d=4)

    for w_test, b_test in [(np.array([1.2, -0.7, 2.5, 0.3]), -0.6),
                           (np.array([-1.0, 0.4, -2.2, 1.8]), 2.1)]:
        yp = predict(X, w_test, b_test)
        dw, db = compute_gradients(X, y, yp)

        assert dw.shape == (4,), f"dw 形状应为 (4,)，实际 {dw.shape}"
        assert np.ndim(db) == 0, f"db 应是标量，实际形状 {np.shape(db)}"

        # w 是向量，要逐个分量扰动（每次只动一个，其余保持不变）
        eps = 1e-6
        dw_num = np.zeros(4)
        for i in range(4):
            wp = w_test.copy(); wp[i] += eps
            wm = w_test.copy(); wm[i] -= eps
            dw_num[i] = (mse_loss(predict(X, wp, b_test), y)
                         - mse_loss(predict(X, wm, b_test), y)) / (2 * eps)

        # b 是标量，直接扰动
        loss_b = lambda b_: mse_loss(predict(X, w_test, b_), y)
        db_num = _numerical_gradient(loss_b, b_test)

        err_w = np.max(np.abs(dw - dw_num) / (np.abs(dw) + np.abs(dw_num) + 1e-12))
        err_b = abs(db - db_num) / (abs(db) + abs(db_num) + 1e-12)
        assert err_w < 1e-6, f"dw 公式错了，最大相对误差 {err_w:.2e}（应 < 1e-6）"
        assert err_b < 1e-6, f"db 公式错了，相对误差 {err_b:.2e}（应 < 1e-6）"


def _check_train():
    X, y, w_true = _make_data(n=200, d=3)
    w, b, losses = train(X, y, lr=0.1, epochs=300)

    assert len(losses) == 300, f"losses 应有 300 个值，实际 {len(losses)}"
    assert losses[-1] < losses[0], "损失没有下降"

    bad = [i for i in range(1, len(losses)) if losses[i] > losses[i - 1] * 1.001]
    assert not bad, f"损失在第 {bad[:5]} 轮出现上升，不是单调下降"

    print(f"      学到  w = {np.round(w, 4)},  b = {b:.4f}")
    print(f"      真实  w = {w_true},  b = 1.5000")
    print(f"      损失  {losses[0]:.4f}  →  {losses[-1]:.6f}")

    assert np.max(np.abs(w - w_true)) < 0.15, "w 离真实值太远"
    assert abs(b - 1.5) < 0.15, "b 离真实值太远"


def main():
    checks = [
        ("1. softmax（数值稳定）", _check_softmax),
        ("2. predict", _check_predict),
        ("3. mse_loss", _check_loss),
        ("4. compute_gradients（梯度检验）", _check_gradients),
        ("5. train（loss 单调下降）", _check_train),
    ]

    passed = 0
    for name, fn in checks:
        try:
            fn()
            print(f"[PASS] {name}")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {name} —— {e}")
        except TypeError as e:
            print(f"[SKIP] {name} —— 函数还没写（{e}）")
        except Exception as e:
            tb = e.__traceback__
            while tb.tb_next:
                tb = tb.tb_next
            print(f"[ERROR] {name} —— 第 {tb.tb_lineno} 行: {type(e).__name__}: {e}")

    print(f"\n{'=' * 50}")
    print(f"通过 {passed} / {len(checks)}")
    print("=" * 50)
    if passed == len(checks):
        print("阶段 0 通关 ✅  可以进入阶段 1（神经网络）了")
    else:
        print("还没通关。哪个没过，就回去【默写】哪部分 —— 别急着往下走。")
        print("阶段 0 是全楼的地基，这里省的时间，后面要加倍还。")


if __name__ == "__main__":
    main()
