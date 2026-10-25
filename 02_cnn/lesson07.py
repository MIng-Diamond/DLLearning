"""
第 07 课：手写卷积和池化的反向传播
=====================================

第 06 课你写了前向。这一课补上反向 —— 训练所需要的梯度。

═══════════════════════════════════════════════════════════════
 前向回顾
═══════════════════════════════════════════════════════════════

    out[co, i, j] = Σ_ci Σ_u Σ_v  x[ci, i·s+u, j·s+v] · W[co, ci, u, v] + b[co]

其中 s 是 stride，u/v 是卷积核内的偏移。

═══════════════════════════════════════════════════════════════
 反向：三个梯度
═══════════════════════════════════════════════════════════════

设 dout = dL/dout，形状和 out 一样 (C_out, out_h, out_w)。

【① 对权重的梯度】

    dW[co, ci, u, v] = Σ_{i,j}  dout[co, i, j] · x[ci, i·s+u, j·s+v]

    直观理解：对每个输出位置 (i,j)，把 dout 在那里的大小，
              乘上输入在该卷积核偏移处 (i·s+u, j·s+v) 的值，累加。

【② 对偏置的梯度】

    db[co] = Σ_{i,j} dout[co, i, j]

【③ 对输入的梯度】

    dx[ci, y, x] += dout[co, i, j] · W[co, ci, u, v]
    其中 y = i·s+u, x = j·s+v

    直观理解：把输出位置的梯度，按卷积核的权重「散射」回输入。
              这是最容易写错的一个 —— 用累加（+=），因为多个输出位置
              会映射到同一个输入位置。

⚠️ 有 padding 时，dx 要在补过零的坐标系里算，最后再把 padding 部分裁掉。

═══════════════════════════════════════════════════════════════
 池化的反向
═══════════════════════════════════════════════════════════════

MaxPool 前向是「每个窗口取最大值」。反向时：

    梯度只回传给那个【最大值所在的位置】，其他位置是 0。

    dx[c, argmax 的行, argmax 的列] += dout[c, i, j]
═══════════════════════════════════════════════════════════════
"""

import numpy as np


# ============================================================
# ① 卷积的反向
# ============================================================
def conv_backward(x, weight, dout, padding=0, stride=1):
    """返回 (dx, dW, db)。

    x      : (C_in, H, W)
    weight : (C_out, C_in, kh, kw)
    dout   : (C_out, out_h, out_w)     上游传来的梯度
    返回
        dx : (C_in, H, W)              和 x 同形状
        dW : (C_out, C_in, kh, kw)     和 weight 同形状
        db : (C_out,)                  和 bias 同形状
    """
    C_in, H, W = x.shape
    C_out, _, kh, kw = weight.shape
    out_h, out_w = dout.shape

    # 和前向一样先补零（注意：只补空间维，不补通道维）
    if padding > 0:
        xp = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    else:
        xp = x

    dxp = np.zeros_like(xp)     # 在补零坐标系里累积 dx
    dW = np.zeros_like(weight)
    db = np.zeros(C_out, dtype=dout.dtype)

    # TODO ①
    # 提示（三重 + 内层）：
    #   for co in range(C_out):
    #       db[co] = np.sum(dout[co])
    #       for ci in range(C_in):
    #           for i in range(out_h):
    #               for j in range(out_w):
    #                   g = dout[co, i, j]
    #                   patch = xp[ci, i*stride:i*stride+kh, j*stride:j*stride+kw]
    #                   dW[co, ci] += g * patch
    #                   dxp[ci, i*stride:i*stride+kh, j*stride:j*stride+kw] += g * weight[co, ci]
    #
    # 注意 dxp 用 += 累加，dW 也是 —— 因为多个位置/通道会贡献到同一处。
    raise NotImplementedError("还没写 conv_backward")

    # 把 padding 部分裁掉
    if padding > 0:
        dx = dxp[:, padding:padding + H, padding:padding + W]
    else:
        dx = dxp
    return dx, dW, db


# ============================================================
# ② 最大池化的反向
# ============================================================
def maxpool_backward(x, dout, size=2, stride=None):
    """返回 dx，形状和 x 相同。

    x    : (C, H, W)      前向时的输入（要用它找最大值的位置）
    dout : (C, out_h, out_w)
    """
    if stride is None:
        stride = size
    C, H, W = x.shape
    out_h, out_w = dout.shape

    dx = np.zeros_like(x, dtype=dout.dtype)

    # TODO ②
    # 提示：
    #   for c in range(C):
    #       for i in range(out_h):
    #           for j in range(out_w):
    #               window = x[c, i*stride:i*stride+size, j*stride:j*stride+size]
    #               # 找最大值在窗口内的二维位置
    #               idx = np.unravel_index(np.argmax(window), window.shape)
    #               dx[c, i*stride + idx[0], j*stride + idx[1]] += dout[c, i, j]
    raise NotImplementedError("还没写 maxpool_backward")


# ============================================================
# 前向（复用第 06 课的思路，这里直接给出，反向检验要用）
# ============================================================
def _conv_single(x, kernel, padding=0, stride=1):
    H, W = x.shape
    kh, kw = kernel.shape
    xp = np.pad(x, ((padding, padding), (padding, padding))) if padding else x
    out_h = (H + 2 * padding - kh) // stride + 1
    out_w = (W + 2 * padding - kw) // stride + 1
    out = np.zeros((out_h, out_w), dtype=x.dtype)
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = np.sum(xp[i*stride:i*stride+kh, j*stride:j*stride+kw] * kernel)
    return out


def conv_forward(x, weight, bias=None, padding=0, stride=1):
    C_in, H, W = x.shape
    C_out, _, kh, kw = weight.shape
    out_h = (H + 2 * padding - kh) // stride + 1
    out_w = (W + 2 * padding - kw) // stride + 1
    out = np.zeros((C_out, out_h, out_w), dtype=x.dtype)
    for co in range(C_out):
        for ci in range(C_in):
            out[co] += _conv_single(x[ci], weight[co, ci], padding, stride)
        if bias is not None:
            out[co] += bias[co]
    return out


def maxpool_forward(x, size=2, stride=None):
    if stride is None:
        stride = size
    C, H, W = x.shape
    out_h = (H - size) // stride + 1
    out_w = (W - size) // stride + 1
    out = np.zeros((C, out_h, out_w), dtype=x.dtype)
    for c in range(C):
        for i in range(out_h):
            for j in range(out_w):
                out[c, i, j] = np.max(
                    x[c, i*stride:i*stride+size, j*stride:j*stride+size])
    return out


# ============================================================
# 梯度检验
# ============================================================
def _rel_err(a, b, eps=1e-12):
    return np.max(np.abs(a - b) / (np.abs(a) + np.abs(b) + eps))


def check_conv_backward():
    rng = np.random.default_rng(0)
    C_in, H, W = 2, 5, 5
    C_out = 2
    kh = kw = 3
    padding, stride = 1, 1

    x = rng.normal(0, 1, (C_in, H, W))
    Wt = rng.normal(0, 1, (C_out, C_in, kh, kw))
    dout = rng.normal(0, 1, (C_out, H, W))     # 假装这是上游梯度

    dx, dW, db = conv_backward(x, Wt, dout, padding, stride)

    # —— 形状 ——
    assert dx.shape == x.shape, f"dx 形状应为 {x.shape}，实际 {dx.shape}"
    assert dW.shape == Wt.shape, f"dW 形状应为 {Wt.shape}，实际 {dW.shape}"
    assert db.shape == (C_out,), f"db 形状应为 ({C_out},)，实际 {db.shape}"

    # —— 数值梯度：定义标量函数 L = Σ(out * dout)，则 dL/dW = dW ——
    def L_of_W(Wt_):
        return np.sum(conv_forward(x, Wt_, padding=padding, stride=stride) * dout)

    def L_of_x(x_):
        return np.sum(conv_forward(x_, Wt, padding=padding, stride=stride) * dout)

    eps = 1e-6

    dW_num = np.zeros_like(Wt)
    it = np.nditer(Wt, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = Wt[idx]
        Wt[idx] = old + eps; lp = L_of_W(Wt)
        Wt[idx] = old - eps; lm = L_of_W(Wt)
        Wt[idx] = old
        dW_num[idx] = (lp - lm) / (2 * eps)
        it.iternext()

    dx_num = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps; lp = L_of_x(x)
        x[idx] = old - eps; lm = L_of_x(x)
        x[idx] = old
        dx_num[idx] = (lp - lm) / (2 * eps)
        it.iternext()

    # db 的数值梯度
    db_num = np.zeros(C_out)
    for co in range(C_out):
        b_plus = np.zeros(C_out); b_plus[co] = eps
        b_minus = np.zeros(C_out); b_minus[co] = -eps
        lp = np.sum(conv_forward(x, Wt, bias=b_plus, padding=padding, stride=stride) * dout)
        lm = np.sum(conv_forward(x, Wt, bias=b_minus, padding=padding, stride=stride) * dout)
        db_num[co] = (lp - lm) / (2 * eps)

    ew, ex, eb = _rel_err(dW, dW_num), _rel_err(dx, dx_num), _rel_err(db, db_num)
    print(f"    dW 相对误差 = {ew:.2e}")
    print(f"    dx 相对误差 = {ex:.2e}")
    print(f"    db 相对误差 = {eb:.2e}")

    assert ew < 1e-6, f"dW 公式错了（相对误差 {ew:.2e}）—— 检查 patch 的取值位置"
    assert ex < 1e-6, f"dx 公式错了（相对误差 {ex:.2e}）—— 检查是否用了 += 累加"
    assert eb < 1e-6, f"db 公式错了（相对误差 {eb:.2e}）"


def check_maxpool_backward():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (2, 6, 6))
    size, stride = 2, 2
    dout = rng.normal(0, 1, (2, 3, 3))

    dx = maxpool_backward(x, dout, size, stride)
    assert dx.shape == x.shape, f"dx 形状应为 {x.shape}，实际 {dx.shape}"

    def L_of_x(x_):
        return np.sum(maxpool_forward(x_, size, stride) * dout)

    eps = 1e-6
    dx_num = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps; lp = L_of_x(x)
        x[idx] = old - eps; lm = L_of_x(x)
        x[idx] = old
        dx_num[idx] = (lp - lm) / (2 * eps)
        it.iternext()

    e = _rel_err(dx, dx_num)
    print(f"    dx 相对误差 = {e:.2e}")
    assert e < 1e-6, f"maxpool 反向错了（相对误差 {e:.2e}）—— 梯度是不是回传给最大值的位置了？"

    # —— 概念检查：每个窗口只有一个位置拿到梯度 ——
    nonzero_per_window = []
    for c in range(2):
        for i in range(3):
            for j in range(3):
                win = dx[c, i*stride:i*stride+size, j*stride:j*stride+size]
                nonzero_per_window.append(np.count_nonzero(win))
    assert all(n == 1 for n in nonzero_per_window), \
        f"每个池化窗口应该只有 1 个位置有梯度，实际 {nonzero_per_window}"
    print("    每个窗口恰好 1 个位置拿到梯度  ✅")


def main():
    checks = [
        ("① 卷积反向（dw / dx / db 梯度检验）", check_conv_backward),
        ("② 最大池化反向（梯度检验）", check_maxpool_backward),
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
            print(f"[ERROR] {name} —— 第 {tb.tb_lineno} 行: {type(e).__name__}: {e}")

    print(f"\n通过 {passed} / {len(checks)}")
    if passed == len(checks):
        print("\n第 07 课完成 ✅  卷积的前向和反向你都亲手写过了。")


if __name__ == "__main__":
    main()
