"""
第 09 课：梯度消失 —— RNN 为什么记不住，LSTM 怎么救的
=========================================================

═══════════════════════════════════════════════════════════════
 一、RNN 的梯度为什么会消失
═══════════════════════════════════════════════════════════════

RNN 的递推：

    h_t = tanh( x_t W_ih^T + h_{t-1} W_hh^T + b )

反向传播时，梯度要从第 T 步一路乘回到第 1 步：

    ∂h_T/∂h_1 = ∏_{t=2}^{T}  ( W_hh^T · diag(1 − tanh²(z_t)) )
                  └──────────┬──────────┘
                       每一步都乘这个

问题在于：
    - tanh 的导数是 1 − tanh² ∈ (0, 1]，通常远小于 1
    - 权重矩阵的元素一般也小于 1
    - 连乘 T 次 → 指数级衰减

T = 100 时，这个乘积通常是 1e-20 甚至 0 —— 梯度彻底消失。

═══════════════════════════════════════════════════════════════
 二、LSTM 的救命通道：细胞状态
═══════════════════════════════════════════════════════════════

LSTM 多了一条【细胞状态】的通道：

    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t

    其中 f_t 是遗忘门（sigmoid 输出，0~1）

看梯度怎么流：

    ∂c_t/∂c_{t-1} = f_t          ← 只乘一个数，不是矩阵，也没有 tanh 导数！

如果 f_t ≈ 1，梯度就能几乎无损地穿过很多步：

    ∂c_T/∂c_1 = f_T · f_{T-1} · ... · f_2 ≈ 1

**这就是 LSTM 能记住长程信息的根本原因。**

═══════════════════════════════════════════════════════════════
 三、实践中的关键技巧：遗忘门 bias 初始化
═══════════════════════════════════════════════════════════════

LSTM 刚初始化时，f_t = sigmoid(0) = 0.5 —— 每步衰减一半，
100 步后 0.5^100 ≈ 8e-31，照样记不住。

所以很多实现会把【遗忘门的 bias 初始化为 1】：

    f_t = sigmoid(1) ≈ 0.73

虽然还是会衰减，但已经好得多，而且训练中有机会把它学得更接近 1。

PyTorch 的 nn.LSTM 默认没有这么做，需要手动设置。

═══════════════════════════════════════════════════════════════
 本课要做的事
═══════════════════════════════════════════════════════════════

直接测量「h0 收到的梯度」随序列长度的变化，对比：
    RNN / GRU / LSTM(默认) / LSTM(遗忘门 bias=1)
"""

import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK HK",
    "WenQuanYi Zen Hei", "Droid Sans Fallback", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# ① 测量 h0 收到的梯度
# ============================================================
def grad_norm_to_h0(cell_type, T, hidden=32, forget_bias=None, seed=0):
    """返回初始隐藏状态 h0 收到的梯度范数。

    cell_type   : "RNN" / "GRU" / "LSTM"
    T           : 序列长度
    forget_bias : 仅对 LSTM 生效；给遗忘门的 bias 设这个值

    做法：
        1. 建模型：C = nn.RNN / nn.GRU / nn.LSTM
        2. 建 h0 = torch.zeros(1, B, hidden, requires_grad=True)
        3. 前向：out, _ = model(x, h0)      （LSTM 要传 (h0, c0)）
        4. loss = out[:, -1].pow(2).sum()   （随便定义一个用到最后一步的标量）
        5. loss.backward()
        6. 返回 h0.grad.norm().item()

    ⚠️ LSTM/GRU 的 bias 布局：
        nn.LSTM 的 bias_ih_l0 形状是 (4*H,)，顺序是 [input, forget, cell, output]
        所以遗忘门是 [H : 2H] 这一段。
        nn.GRU 是 (3*H,)，顺序是 [reset, update, new]。
    """
    # TODO
    torch.manual_seed(seed)
    C = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[cell_type]
    model = C(1, hidden, batch_first=True)
    if cell_type == "LSTM" and forget_bias is not None:
        set_forget_bias(model, forget_bias)

    x = torch.randn(2, T, 1)
    h0 = torch.zeros(1, 2, hidden, requires_grad=True)
    if cell_type == "LSTM":
        out, _ = model(x, (h0, torch.zeros(1, 2, hidden)))
    else:
        out, _ = model(x, h0)

    out[:, -1].pow(2).sum().backward()
    return h0.grad.norm().item()


# ============================================================
# ② 设置 LSTM 的遗忘门 bias
# ============================================================
def set_forget_bias(model, value):
    """把 LSTM 的遗忘门 bias 设成 value。

    nn.LSTM 有：
        model.bias_ih_l0 : (4*H,)   [input, forget, cell, output]
        model.bias_hh_l0 : (4*H,)

    两处都要设（PyTorch 的 LSTM 把 bias 拆成两份，最后会相加）。
    """
    # TODO
    H = model.hidden_size
    with torch.no_grad():
        model.bias_ih_l0[H:2 * H].fill_(value)
        model.bias_hh_l0[H:2 * H].fill_(value)


# ============================================================
# 画图（已写好）
# ============================================================
def plot_results(ts, series, path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "lesson09_result.png")

    colors = {
        "RNN": "#e34948",
        "GRU": "#eb6834",
        "LSTM(默认)": "#898781",
        "LSTM(遗忘门bias=1)": "#2a78d6",
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for name, vals in series.items():
        safe = [max(v, 1e-40) for v in vals]      # 避免 log(0)
        ax.plot(ts, safe, "o-", linewidth=2, markersize=5,
                color=colors.get(name, "#333333"), label=name)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("序列长度 T")
    ax.set_ylabel("h0 收到的梯度范数（对数刻度）")
    ax.set_title("梯度能传多远？  RNN vs GRU vs LSTM")
    ax.grid(color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(path, dpi=130, facecolor="#fcfcfb")
    print(f"    图片已保存: {path}")


# ============================================================
# 主流程
# ============================================================
def main():
    checks = [
        ("① 梯度测量", check_grad),
        ("② 遗忘门 bias", check_bias),
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
        ts = [5, 10, 20, 50, 100, 200]
        series = {
            "RNN": [grad_norm_to_h0("RNN", t) for t in ts],
            "GRU": [grad_norm_to_h0("GRU", t) for t in ts],
            "LSTM(默认)": [grad_norm_to_h0("LSTM", t) for t in ts],
            "LSTM(遗忘门bias=1)": [
                grad_norm_to_h0("LSTM", t, forget_bias=1.0) for t in ts
            ],
        }

        print()
        print(f"{'T':>5} | " + " | ".join(f"{k:>18}" for k in series))
        print("-" * 88)
        for i, t in enumerate(ts):
            print(f"{t:>5} | " + " | ".join(
                f"{series[k][i]:>18.2e}" for k in series))

        plot_results(ts, series)
        print("\n第 09 课完成 ✅")
        print("看图：梯度随 T 的衰减速度，LSTM(bias=1) 明显最平缓。")


# ============================================================
# 检查
# ============================================================
def check_grad():
    vals = [grad_norm_to_h0("RNN", t) for t in (5, 50)]
    assert vals[0] > vals[1], \
        f"T=5 的梯度应大于 T=50（梯度在衰减），实际 {vals[0]:.2e} vs {vals[1]:.2e}"
    assert vals[0] > 1e-6, f"T=5 的梯度太小（{vals[0]:.2e}），检查 h0 是否传对了"
    print(f"    RNN: T=5 → {vals[0]:.2e},  T=50 → {vals[1]:.2e}  （确实在衰减）")


def check_bias():
    torch.manual_seed(0)
    m = nn.LSTM(1, 8, batch_first=True)
    set_forget_bias(m, 1.0)
    H = 8
    assert torch.allclose(m.bias_ih_l0[H:2*H], torch.ones(H)), \
        f"bias_ih_l0 的遗忘门部分应全为 1，实际 {m.bias_ih_l0[H:2*H]}"
    assert torch.allclose(m.bias_hh_l0[H:2*H], torch.ones(H)), \
        f"bias_hh_l0 的遗忘门部分应全为 1，实际 {m.bias_hh_l0[H:2*H]}"
    print(f"    forget gate bias 已设为 1（两处 bias）  ✅")


if __name__ == "__main__":
    main()
