"""
第 06 课：手写 2D 卷积
========================

【目标】不用 torch.nn.Conv2d / F.conv2d，自己用循环实现卷积，
        然后验证结果和 PyTorch 的官方实现【逐位一致】。

═══════════════════════════════════════════════════════════════
 卷积到底在算什么？
═══════════════════════════════════════════════════════════════

拿一个 3×3 的「核」（kernel / filter），在图上从左到右、从上到下地滑：

    原图 5×5                    核 3×3
    ┌─────────────┐            ┌───┬───┬───┐
    │ 1  2  3  0  1│            │ 1 │ 0 │-1 │
    │ 0  1  2  3  0│      ×     ├───┼───┼───┤
    │ 1  0  1  2  3│            │ 2 │ 0 │-2 │
    │ 2  1  0  1  2│            ├───┼───┼───┤
    │ 3  2  1  0  1│            │ 1 │ 0 │-1 │
    └─────────────┘            └───┴───┴───┘

在左上角位置，把核盖上去，【对应位置相乘再全部加起来】：

    1×1 + 2×0 + 3×(-1)
  + 0×2 + 1×0 + 2×(-2)   =  -8
  + 1×1 + 0×0 + 1×(-1)

这个 -8 就是输出图左上角的第一个值。然后窗口往右滑一格，重复。

【关键认知】
  1. 卷积 = 「局部窗口的点积」+「滑动」
  2. 同一个核扫全图 → 权重共享 → 参数量少
  3. 不同的核检测不同的模式（这个核检测的是「竖直边缘」）

═══════════════════════════════════════════════════════════════
 输出尺寸公式（写 CNN 必须背下来）
═══════════════════════════════════════════════════════════════

    out = ( (in + 2 × padding − kernel) // stride ) + 1

    padding=1, kernel=3, stride=1  →  out = in     （尺寸不变）
    kernel=2, stride=2             →  out = in/2   （尺寸减半，池化常用）
═══════════════════════════════════════════════════════════════
"""

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# ① 单通道卷积（先写这个，最简单）
# ============================================================
def conv2d_single(x, kernel, padding=0, stride=1):
    """单通道 2D 卷积。

    x      : (H, W)          一张灰度图
    kernel : (kh, kw)        一个卷积核
    返回   : (out_h, out_w)

    步骤：
      1. 如果 padding > 0，用 np.pad 在四周补 0
             np.pad(x, ((p, p), (p, p)))
      2. 算输出尺寸：
             out_h = (H + 2*padding - kh) // stride + 1
             out_w = (W + 2*padding - kw) // stride + 1
      3. 造一个 (out_h, out_w) 的全零输出
      4. 四重循环（其实只要两重循环遍历输出位置）：
             for i in range(out_h):
                 for j in range(out_w):
                     取出窗口 = xp[i*stride : i*stride+kh,
                                  j*stride : j*stride+kw]
                     out[i, j] = np.sum(窗口 * kernel)
    """
    # TODO ①
    x = np.pad(x, ((padding, padding), (padding, padding)))
    out_h = (x.shape[0] - kernel.shape[0]) // stride + 1
    out_w = (x.shape[1] - kernel.shape[1]) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            tmp = x[i * stride : i * stride + kernel.shape[0],
                    j * stride : j * stride + kernel.shape[1]]
            out[i, j] = np.sum(tmp * kernel)
    return out
    raise NotImplementedError("还没写 conv2d_single")


# ============================================================
# ② 多通道、多卷积核（真正在 CNN 里用的版本）
# ============================================================
def conv2d(x, weight, bias=None, padding=0, stride=1):
    """多通道卷积。

    x      : (C_in, H, W)              比如 1 张灰度图是 (1,28,28)，RGB 是 (3,H,W)
    weight : (C_out, C_in, kh, kw)     一共 C_out 个卷积核，每个核有 C_in 层
    bias   : (C_out,) 或 None
    返回   : (C_out, out_h, out_w)

    【怎么算】
      对每一个输出通道 co：
          先置零
          对每一个输入通道 ci：
              out[co] += conv2d_single(x[ci], weight[co, ci], padding, stride)
          最后 out[co] += bias[co]

    也就是说：一个输出通道 = 把所有输入通道的卷积结果【加起来】。

    提示：
      - 返回的 shape 用 conv2d_single 算一次就能知道
      - 一定要传 padding / stride，不然形状对不上
    """
    # TODO ②
    x = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    out_h = (x.shape[1] - weight.shape[2]) // stride + 1
    out_w = (x.shape[2] - weight.shape[3]) // stride + 1

    out = np.zeros((weight.shape[0], out_h, out_w))
    for co in range(weight.shape[0]):
        for ci in range(x.shape[0]):
            out[co] += conv2d_single(x[ci], weight[co, ci], 0, stride=stride)
        if bias is not None:
            out[co] += bias[co]
    return out
    raise NotImplementedError("还没写 conv2d")

# ============================================================
# ③ 最大池化
# ============================================================
def maxpool2d(x, size=2, stride=None):
    """最大池化：每个 size×size 窗口里取最大值。

    x      : (C, H, W)
    size   : 窗口大小（默认 2）
    stride : 步长（默认等于 size，所以尺寸正好减半）
    返回   : (C, out_h, out_w)

    步骤：
      out_h = (H - size) // stride + 1
      out_w = (W - size) // stride + 1
      三重循环（通道、行、列），每个窗口取 np.max
    """
    # TODO ③
    if stride is None:
        stride = size
    out_h = (x.shape[1] - size) // stride + 1
    out_w = (x.shape[2] - size) // stride + 1
    out = np.zeros((x.shape[0], out_h, out_w))
    for c in range(x.shape[0]):
        for i in range(out_h):
            for j in range(out_w):
                out[c, i, j] = np.max(x[c, i * stride : i * stride + size,
                                         j * stride : j * stride + size])
    return out
    raise NotImplementedError("还没写 maxpool2d")


# ============================================================
# 检查（和 PyTorch 官方实现逐位对比）
# ============================================================
def _np2t(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32)


def check_single():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (7, 8)).astype(np.float32)

    for kh, kw, p, s in [(3, 3, 0, 1), (3, 3, 1, 1), (5, 5, 2, 1),
                         (3, 3, 1, 2), (2, 2, 0, 2)]:
        k = rng.normal(0, 1, (kh, kw)).astype(np.float32)
        got = conv2d_single(x, k, padding=p, stride=s)
        want = F.conv2d(_np2t(x)[None, None], _np2t(k)[None, None],
                        padding=p, stride=s).numpy()[0, 0]
        assert got.shape == want.shape, \
            f"kernel={kh}x{kw}, padding={p}, stride={s}: " \
            f"形状应为 {want.shape}，实际 {got.shape}"
        assert np.allclose(got, want, atol=1e-5), \
            f"kernel={kh}x{kw}, padding={p}, stride={s}: 数值不一致\n" \
            f"你算的:\n{np.round(got, 3)}\nPyTorch:\n{np.round(want, 3)}"

    print("    5 组 (kernel/padding/stride) 组合全部与 PyTorch 一致  ✅")


def check_multi():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (3, 9, 11)).astype(np.float32)
    w = rng.normal(0, 0.2, (4, 3, 3, 3)).astype(np.float32)
    b = rng.normal(0, 0.1, 4).astype(np.float32)

    got = conv2d(x, w, bias=b, padding=1, stride=1)
    want = F.conv2d(_np2t(x)[None], _np2t(w), bias=_np2t(b),
                    padding=1, stride=1).numpy()[0]

    assert got.shape == want.shape, f"形状应为 {want.shape}，实际 {got.shape}"
    assert np.allclose(got, want, atol=1e-5), \
        f"数值不一致，最大差 {np.abs(got - want).max():.2e}"

    # 不加 bias 也要对
    got2 = conv2d(x, w, padding=1, stride=1)
    want2 = F.conv2d(_np2t(x)[None], _np2t(w), padding=1, stride=1).numpy()[0]
    assert np.allclose(got2, want2, atol=1e-5), "不带 bias 时不一致"

    print(f"    输入 {x.shape} × 权重 {w.shape} → 输出 {got.shape}，与 PyTorch 一致  ✅")


def check_pool():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, (2, 8, 10)).astype(np.float32)

    got = maxpool2d(x, size=2)
    want = F.max_pool2d(_np2t(x)[None], kernel_size=2).numpy()[0]
    assert got.shape == want.shape, f"形状应为 {want.shape}，实际 {got.shape}"
    assert np.allclose(got, want), "maxpool 数值不一致"

    got3 = maxpool2d(x, size=3, stride=2)
    want3 = F.max_pool2d(_np2t(x)[None], kernel_size=3, stride=2).numpy()[0]
    assert np.allclose(got3, want3), "size=3, stride=2 时不一致"

    print(f"    输入 {x.shape} → 输出 {got.shape}，与 PyTorch 一致  ✅")


def check_shapes():
    """手动验证尺寸公式"""
    assert conv2d_single(np.zeros((28, 28)), np.zeros((3, 3)), padding=1).shape == (28, 28), \
        "3×3 + padding=1 应该保持尺寸不变"
    assert conv2d_single(np.zeros((28, 28)), np.zeros((3, 3))).shape == (26, 26), \
        "3×3 不加 padding 应该缩小 2"
    assert conv2d_single(np.zeros((28, 28)), np.zeros((3, 3)), padding=1, stride=2).shape == (14, 14), \
        "stride=2 应该减半"
    assert maxpool2d(np.zeros((1, 28, 28))).shape == (1, 14, 14), \
        "2×2 池化应该减半"
    print("    尺寸公式全部正确  ✅")


def main():
    checks = [
        ("尺寸公式", check_shapes),
        ("① 单通道卷积", check_single),
        ("② 多通道卷积", check_multi),
        ("③ 最大池化", check_pool),
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
        print("\n第 06 课完成 ✅  你现在亲手实现了卷积 —— nn.Conv2d 不再是黑盒。")


if __name__ == "__main__":
    main()
