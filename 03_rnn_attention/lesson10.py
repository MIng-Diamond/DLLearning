"""
第 10 课：Attention —— 一步看到所有位置
==========================================

【为什么需要 Attention】

RNN/LSTM 的信息必须沿着序列一步步传：
    h1 → h2 → h3 → ... → h100
距离越远，信息越模糊。

Attention 换了个思路：
    【每个位置直接和所有位置计算相关度，一步到位】

第 1 个位置和第 100 个位置之间，只隔了一次点积，没有任何衰减。

═══════════════════════════════════════════════════════════════
 核心公式
═══════════════════════════════════════════════════════════════

    Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V

    Q : (B, T_q, D)     Query  "我在找什么"
    K : (B, T_k, D)     Key    "我有什么标签"
    V : (B, T_k, D_v)   Value  "我的内容是什么"

    一步一步：
        ① scores = Q @ K.transpose(-1, -2)      → (B, T_q, T_k)  相似度
        ② scores = scores / sqrt(D)             → 缩放
        ③ attn   = softmax(scores, dim=-1)      → (B, T_q, T_k)  每行和为 1
        ④ out    = attn @ V                     → (B, T_q, D_v)  加权混合

【为什么要除以 √d_k？】
    点积的方差随维度 D 增长。D 很大时，scores 的数值会很大，
    经过 softmax 后会变成"一个 1 其余全 0"（饱和），梯度就没了。
    除以 √D 把方差拉回 1 左右。

═══════════════════════════════════════════════════════════════
 Multi-Head Attention（多头注意力）
═══════════════════════════════════════════════════════════════

单个注意力只能学一种"关注模式"。多头让它同时学多种：

    head_i = Attention(Q·W_q_i, K·W_k_i, V·W_v_i)
    out    = Concat(head_1, ..., head_h) · W_o

    把 D 维切成 h 份，每份 D/h 维，各算各的注意力，最后拼起来。

    比喻：一个头关注"语法关系"，另一个头关注"指代关系"，
          再一个关注"位置相邻"……最后综合。
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK HK",
    "WenQuanYi Zen Hei", "Droid Sans Fallback", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# ① 缩放点积注意力（本课核心）
# ============================================================
def scaled_dot_product_attention(q, k, v, mask=None):
    """返回 (out, attn)。

    q    : (B, T_q, D)
    k    : (B, T_k, D)
    v    : (B, T_k, D_v)
    mask : (B, T_q, T_k) 或 None。mask 里为 0 的位置不许被关注。

    返回
        out  : (B, T_q, D_v)
        attn : (B, T_q, T_k)   注意力权重（每行和为 1）

    步骤（四行）：
        1. scores = q @ k.transpose(-1, -2)          # (B,Tq,Tk)
        2. scores = scores / math.sqrt(q.size(-1))   # 缩放
        3. 如果有 mask：scores = scores.masked_fill(mask == 0, float("-inf"))
        4. attn = torch.softmax(scores, dim=-1)
        5. return attn @ v, attn
    """
    # TODO ①
    B = q.shape[0]
    D = q.shape[-1]
    scores = q @ k.transpose(-1, -2)
    scores = scores / math.sqrt(D)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(scores, dim = -1)
    out = attn @ v
    return (out, attn)
    raise NotImplementedError("还没写 scaled_dot_product_attention")


# ============================================================
# ② 多头注意力
# ============================================================
def multi_head_attention(q, k, v, W_q, W_k, W_v, W_o, n_heads):
    """返回 (out, attn_avg)。

    q: (B, T_q, D)   k: (B, T_k, D)   v: (B, T_k, D)
    W_q, W_k, W_v : nn.Linear(D, D)
    W_o           : nn.Linear(D, D)
    n_heads       : 头数（D 必须能被它整除）

    ⚠️ 最常见的实现方式：用【一个】Linear(D, D) 一次投影，
       然后 reshape 成多头。这样比定义 h 个 Linear 高效。

    步骤：
        1. 投影：
             q = W_q(q)   # (B, Tq, D)
             k = W_k(k)
             v = W_v(v)
        2. 拆成多头：
             B, Tq, D = q.shape
             hd = D // n_heads
             q = q.view(B, Tq, n_heads, hd).transpose(1, 2)   # (B, h, Tq, hd)
             对 k / v 同样处理（注意 T_k）
        3. 对每个头做 attention：
             out, attn = scaled_dot_product_attention(q, k, v)
             # out: (B, h, Tq, hd)
        4. 拼回：
             out = out.transpose(1, 2).reshape(B, Tq, D)
        5. 输出投影：
             out = W_o(out)
        6. 返回 (out, attn_avg)，attn_avg 是各头权重的平均 (B, Tq, Tk)
    """
    # TODO ②
    q = W_q(q)
    k = W_k(k)
    v = W_v(v)
    B, Tq, D = q.shape
    Tk = k.shape[1]
    Tv = v.shape[1]
    hd = D // n_heads
    q = q.reshape(B, Tq, n_heads, hd).transpose(1, 2)
    k = k.reshape(B, Tk, n_heads, hd).transpose(1, 2)
    v = v.reshape(B, Tv, n_heads, hd).transpose(1, 2)
    out, attn = scaled_dot_product_attention(q, k, v)
    out = W_o(out.transpose(1, 2).reshape(B, Tq, D))
    attn_avg = attn.mean(dim = 1)
    return (out, attn_avg)
    raise NotImplementedError("还没写 multi_head_attention")


# ============================================================
# 可视化（已写好）
# ============================================================
def plot_attention(attn, tokens, path=None, title="注意力权重"):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "lesson10_result.png")
    a = attn.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    fig.patch.set_facecolor("#fcfcfb")
    im = ax.imshow(a, cmap="Blues", vmin=0)
    ax.set_xlabel("被关注的 Key 位置")
    ax.set_ylabel("发起查询的 Query 位置")
    ax.set_title(title)
    if tokens is not None:
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)
    fig.colorbar(im, ax=ax, label="权重")
    plt.tight_layout()
    plt.savefig(path, dpi=130, facecolor="#fcfcfb")
    print(f"    图片已保存: {path}")


# ============================================================
# 检查
# ============================================================
def check_attention():
    torch.manual_seed(0)
    B, Tq, Tk, D, Dv = 2, 4, 6, 8, 5

    q = torch.randn(B, Tq, D)
    k = torch.randn(B, Tk, D)
    v = torch.randn(B, Tk, Dv)

    out, attn = scaled_dot_product_attention(q, k, v)

    assert out.shape == (B, Tq, Dv), f"out 应为 {(B,Tq,Dv)}，实际 {tuple(out.shape)}"
    assert attn.shape == (B, Tq, Tk), f"attn 应为 {(B,Tq,Tk)}，实际 {tuple(attn.shape)}"
    assert torch.allclose(attn.sum(-1), torch.ones(B, Tq), atol=1e-6), \
        "attn 的每一行加起来应为 1"

    # 和 PyTorch 官方实现对比
    ref = F.scaled_dot_product_attention(q, k, v)
    err = float((out - ref).abs().max())
    assert err < 1e-6, f"和 F.scaled_dot_product_attention 不一致，最大差 {err:.2e}"
    print(f"    输出 {tuple(out.shape)}，权重 {tuple(attn.shape)}，与 PyTorch 一致  ✅")

    # mask 测试
    mask = torch.ones(B, Tq, Tk)
    mask[:, :, 2:] = 0
    _, attn_m = scaled_dot_product_attention(q, k, v, mask=mask)
    assert torch.allclose(attn_m[:, :, 2:], torch.zeros_like(attn_m[:, :, 2:])), \
        "被 mask 掉的位置权重应为 0"
    print("    mask 生效：被屏蔽的位置权重为 0  ✅")


def check_multihead():
    torch.manual_seed(0)
    B, T, D, h = 2, 5, 8, 2
    q = k = v = torch.randn(B, T, D)

    W_q, W_k, W_v, W_o = [nn.Linear(D, D) for _ in range(4)]
    out, attn = multi_head_attention(q, k, v, W_q, W_k, W_v, W_o, n_heads=h)

    assert out.shape == (B, T, D), f"out 应为 {(B,T,D)}，实际 {tuple(out.shape)}"
    assert attn.shape == (B, T, T), f"attn 应为 {(B,T,T)}，实际 {tuple(attn.shape)}"
    assert torch.allclose(attn.sum(-1), torch.ones(B, T), atol=1e-6), \
        "attn 每行和应为 1"
    print(f"    多头输出 {tuple(out.shape)}，权重 {tuple(attn.shape)}  ✅")


def main():
    checks = [
        ("① 缩放点积注意力", check_attention),
        ("② 多头注意力", check_multihead),
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
        # 用一个正弦序列演示注意力权重
        torch.manual_seed(0)
        T, D = 8, 16
        x = torch.sin(torch.arange(T).float() * 0.8)[:, None] * torch.randn(D)[None, :]
        x = x[None]                                   # (1, T, D)
        W_q, W_k, W_v, W_o = [nn.Linear(D, D) for _ in range(4)]
        _, attn = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, n_heads=2)
        plot_attention(attn[0], [f"t{i}" for i in range(T)],
                       title="自注意力权重（每一行 = 一个位置在看谁）")
        print("\n第 10 课完成 ✅  Attention 是 Transformer 的核心")


if __name__ == "__main__":
    main()
