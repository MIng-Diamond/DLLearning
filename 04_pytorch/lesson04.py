"""
第 04 课：用 PyTorch 重写 —— 看 autograd 怎么替代你手推的梯度
================================================================

这一课很短，但很重要。你会看到三件事：

  1. PyTorch 的张量操作和你学的 NumPy 几乎一模一样（.shape / @ / mean）
  2. 你手推的 compute_gradients 整段，被 loss.backward() 一行替代
  3. 【最关键】torch 算出来的梯度，和你手推的公式算出的是同一个数

第 3 点是这一课的意义：它把 autograd 从「黑盒」变成「透明的」。

═══════════════════════════════════════════════════════════════
 PyTorch 的三个新概念
═══════════════════════════════════════════════════════════════

【requires_grad=True】
    告诉 torch：「这个张量是需要求导的参数，请盯着它的一举一动」。
    torch 会因此记录下所有和它相关的运算，构建一张「计算图」。

【loss.backward()】
    沿着计算图往回走，自动算出 loss 对每个 requires_grad 张量的梯度，
    结果存在 .grad 属性里。这就是「反向传播」。

【.grad.zero_()】
    ⚠️ torch 的梯度是【累加】的，不是覆盖！不清零的话，下一次 backward
    会加在上一次的结果上，越滚越大。这是 PyTorch 初学者的第一大坑。

═══════════════════════════════════════════════════════════════
"""

import sys
import numpy as np
import torch


# ============================================================
# 数据（和第 03 课完全一样，只是后面转成 torch 张量）
# ============================================================
def make_data(n=200, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d))
    w_true = np.array([2.0, -3.0, 0.5])
    b_true = 1.5
    y = X @ w_true + b_true + rng.normal(0, 0.3, n)
    return X, y, w_true, b_true


# ============================================================
# ① NumPy 数组 → torch 张量
# ============================================================
def to_tensor(arr, device="cpu"):
    """把 numpy 数组变成 torch 张量。

    提示：torch.tensor(arr, dtype=torch.float32, device=device)

    ⚠️ dtype 必须写 float32！torch 的神经网络运算默认用 float32，
       而 numpy 默认是 float64，不转会报类型错误。
    """
    # TODO ①
    return torch.tensor(arr, dtype = torch.float32, device = device)

    raise NotImplementedError("还没写 to_tensor")


def make_params(d, device="cpu"):
    """创建需要求导的参数 w 和 b。

    提示：
        w = torch.zeros(d, device=device, requires_grad=True)
        b = torch.zeros(1, device=device, requires_grad=True)

    b 用形状 (1,) 是为了能广播（(N,) 和 (1,) 相加 → (N,)）。
    """
    # TODO ②
    w = torch.zeros(d, device = device, requires_grad = True)
    b = torch.zeros(1, device = device, requires_grad = True)
    return w, b
    raise NotImplementedError("还没写 make_params")


# ============================================================
# ③ 训练循环 —— 四步循环，但第③步变成了一行
# ============================================================
def train_torch(X, y, lr=0.1, epochs=300):
    """用 PyTorch 训练，返回 (w, b, losses)"""
    # 注意 device=X.device：如果 X 在 GPU 上，参数也必须建在 GPU 上
    w, b = make_params(X.shape[1], device=X.device)

    losses = []
    for epoch in range(epochs):
        # ① 前向 —— 和你写的 predict 一模一样
        y_pred = X @ w + b

        # ② 损失 —— 和你写的 mse_loss 一模一样
        loss = ((y_pred - y) ** 2).mean()
        losses.append(loss.item())      # .item() 把单元素张量变成 python 数字

        # ③ 反向 —— 【这一行，替代了你整个 compute_gradients】
        # TODO ③：调用 backward
        loss.backward()
        # ④ 更新 —— 必须放在 torch.no_grad() 里！
        #    否则这次减法也会被记进计算图，越滚越大，最后爆内存。
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        # ⑤ 清空梯度 —— 必须！torch 的梯度是累加的
        # TODO ④：把 w.grad 和 b.grad 清零
        w.grad.zero_()
        b.grad.zero_()
        #    提示：张量有一个原地操作 .zero_()（注意结尾的下划线）

    return w.detach(), b.detach(), losses


# ============================================================
# ⑤ 对比：autograd 算的 vs 你手推的公式（本课的重头戏）
# ============================================================
def compare_gradients(X, y, w, b):
    """证明 torch 的 autograd 和你手推的公式，算出来是同一个数。（已写好）"""
    # —— 路线 A：让 torch 自动求导 ——
    y_pred = X @ w + b
    loss = ((y_pred - y) ** 2).mean()
    loss.backward()                          # autograd 干活
    dw_torch = w.grad.detach().numpy()
    db_torch = b.grad.detach().numpy()

    # —— 路线 B：用你第 03 课手推的公式（纯 numpy）——
    X_np = X.detach().numpy()
    y_np = y.detach().numpy()
    w_np = w.detach().numpy()
    b_np = b.detach().numpy()
    d = (X_np @ w_np + b_np) - y_np          # 误差
    N = len(y_np)
    dw_manual = (2 / N) * (X_np.T @ d)       # ← 你手推的公式
    db_manual = (2 / N) * np.sum(d)

    print("  路线A autograd  dw =", np.round(dw_torch, 8))
    print("  路线B 你手推的  dw =", np.round(dw_manual, 8))
    print("  路线A autograd  db =", np.round(db_torch, 8))
    print("  路线B 你手推的  db =", np.round(db_manual, 8))
    print()
    err_w = np.max(np.abs(dw_torch - dw_manual))
    err_b = np.max(np.abs(db_torch - db_manual))
    print(f"  最大差异:  dw {err_w:.2e}   db {err_b:.2e}   ← float32 精度内就是同一个数")

    assert err_w < 1e-5, f"autograd 和手推公式对不上！差异 {err_w:.2e}"
    assert err_b < 1e-5, f"autograd 和手推公式对不上！差异 {err_b:.2e}"

    # 清理梯度，免得影响后面的调用
    w.grad.zero_()
    b.grad.zero_()


# ============================================================
# ⑥ 演示：为什么必须 zero_grad（已写好，直接看输出）
# ============================================================
def demo_zero_grad():
    """一个小实验：不清零梯度会发生什么"""
    print("  设 x = 3.0，函数是 x²，那么真实梯度 = 2x = 6")
    print()
    x = torch.tensor(3.0, requires_grad=True)
    for i in range(3):
        loss = x ** 2
        loss.backward()
        print(f"    第 {i+1} 次 backward 后，x.grad = {x.grad.item():.1f}"
              + ("   ← 应该是 6 才对！" if i > 0 else ""))

    print()
    print("  看到没：6 → 12 → 18，梯度在【累加】而不是被覆盖。")
    print("  所以每次更新完参数，必须手动 .zero_() 清空。")


# ============================================================
# 检查
# ============================================================
def check_to_tensor():
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    t = to_tensor(a)
    assert isinstance(t, torch.Tensor), f"应返回 torch.Tensor，实际 {type(t)}"
    assert t.shape == (2, 3), f"形状应为 (2,3)，实际 {t.shape}"
    assert t.dtype == torch.float32, f"dtype 应为 float32，实际 {t.dtype}"
    assert np.allclose(t.numpy(), a), "数值应该和原数组一致"


def check_params():
    w, b = make_params(4)
    assert w.shape == (4,), f"w 形状应为 (4,)，实际 {w.shape}"
    assert w.requires_grad and b.requires_grad, "requires_grad 必须是 True，否则求不了梯度"


def check_train():
    X_np, y_np, w_true, b_true = make_data()
    X = to_tensor(X_np)
    y = to_tensor(y_np)
    w, b, losses = train_torch(X, y, lr=0.1, epochs=300)

    assert len(losses) == 300, f"losses 应有 300 个，实际 {len(losses)}"
    assert losses[-1] < losses[0], "损失没有下降"
    assert losses[-1] < 0.2, f"最终损失 {losses[-1]:.4f} 太大，应 < 0.2"

    w_np = w.numpy()
    print(f"    学到:  w = {np.round(w_np, 4)},  b = {b.item():.4f}")
    print(f"    真实:  w = {w_true},  b = {b_true}")
    assert np.max(np.abs(w_np - w_true)) < 0.15, "学到的 w 离真实值太远"


def check_compare():
    X_np, y_np, _, _ = make_data()
    X, y = to_tensor(X_np), to_tensor(y_np)
    w = torch.tensor([1.2, -0.7, 2.5], requires_grad=True)
    b = torch.tensor([0.3], requires_grad=True)
    compare_gradients(X, y, w, b)


def main():
    checks = [
        ("① numpy → torch 张量", check_to_tensor),
        ("② 创建可求导参数 make_params", check_params),
        ("③ 训练循环 train_torch", check_train),
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
        print()
        print("=" * 62)
        print("autograd vs 你手推的公式 —— 直接对比")
        print("=" * 62)
        check_compare()
        print()
        print("=" * 62)
        print("为什么必须 zero_grad")
        print("=" * 62)
        demo_zero_grad()
        print()
        print("=" * 62)
        print("GPU 试试（你的 RTX 3060）")
        print("=" * 62)
        if torch.cuda.is_available():
            dev = "cuda"
            X = to_tensor(make_data()[0], device=dev)
            y = to_tensor(make_data()[1], device=dev)
            w, b, losses = train_torch(X, y)
            print(f"  在 {torch.cuda.get_device_name(0)} 上训练完成，最终损失 {losses[-1]:.6f}")
            print(f"  张量所在设备: {w.device}")
        else:
            print("  CUDA 不可用")


if __name__ == "__main__":
    main()
