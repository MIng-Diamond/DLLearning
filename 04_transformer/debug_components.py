"""
逐组件调试：从小零件到大模型
================================

用法：
    cd 04_transformer
    python debug_components.py

它会按依赖顺序，一个一个测试组件。
第一个 [FAIL] 就是你要先修的地方。

调试方法：
    1. 每个组件用【极小的尺寸】(B=2, T=3, D=8)
    2. 每一步都 assert 形状
    3. 一个组件过了再测下一个
    4. 不要一次改多个地方
"""

import traceback
import torch

from lesson11 import (
    PositionalEncoding, MultiHeadAttention, FeedForward,
    TransformerBlock, MiniGPT,
)

D = 8          # 极小维度，方便看清
B, T = 2, 3    # 极小 batch 和序列长度
V = 10         # 极小词表


def run(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}")
        print(f"       {type(e).__name__}: {e}")
        # 打印完整 traceback，让你看到出错的行
        tb = traceback.format_exc().splitlines()
        for line in tb[-6:]:
            print("       " + line)
        return False


# ============================================================
def test_pos_enc():
    pe = PositionalEncoding(D=D, max_len=16)
    x = torch.randn(B, T, D)
    out = pe(x)
    assert out.shape == (B, T, D), f"期望 {(B,T,D)}，实际 {tuple(out.shape)}"
    # 位置编码应该是【加】上去的，不是乘
    assert not torch.allclose(out, x * x), "输出看起来像乘法"


def test_mha():
    mha = MultiHeadAttention(D=D, n_heads=2)
    x = torch.randn(B, T, D)

    out = mha(x)
    assert out.shape == (B, T, D), f"期望 {(B,T,D)}，实际 {tuple(out.shape)}"

    # 多头：内部应该拆成 2 个头
    # 用 causal mask 也测一次
    mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    out2 = mha(x, attn_mask=mask)
    assert out2.shape == (B, T, D)


def test_ffn():
    ffn = FeedForward(D=D)
    x = torch.randn(B, T, D)
    out = ffn(x)
    assert out.shape == (B, T, D), f"期望 {(B,T,D)}，实际 {tuple(out.shape)}"


def test_block():
    block = TransformerBlock(D=D, n_heads=2)
    x = torch.randn(B, T, D)
    mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    out = block(x, mask)
    assert out.shape == (B, T, D), f"期望 {(B,T,D)}，实际 {tuple(out.shape)}"


def test_model():
    model = MiniGPT(vocab_size=V, D=D, n_heads=2, n_layers=2, max_len=16)
    idx = torch.randint(0, V, (B, T))
    mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    out = model(idx, mask)
    assert out.shape == (B, T, V), f"期望 {(B,T,V)}，实际 {tuple(out.shape)}"

    # 输出应该是 logits（不是概率）：各行和不应为 1
    row_sums = out.softmax(-1).sum(-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_training_step():
    """只跑一步训练，验证 train_model 能不能动。"""
    from lesson11 import build_dataset, causal_mask, train_model

    text = "hello world this is a tiny test text for debugging"
    get_batch, stoi, itos, vocab = build_dataset(text, block_size=8)
    model = MiniGPT(vocab_size=vocab, D=D, n_heads=2, n_layers=1, max_len=16)

    losses = train_model(model, get_batch, "cpu", steps=2, log_every=1)
    assert len(losses) == 2, f"losses 应有 2 个，实际 {len(losses)}"
    assert all(isinstance(x, float) for x in losses), "losses 里应该是 float，不是 tensor"


def test_generate():
    from lesson11 import build_dataset, generate

    text = "hello world this is a tiny test text"
    get_batch, stoi, itos, vocab = build_dataset(text, block_size=8)
    model = MiniGPT(vocab_size=vocab, D=D, n_heads=2, n_layers=1, max_len=16)
    out = generate(model, "he", stoi, itos, "cpu", n_chars=10, top_k=5)
    assert isinstance(out, str), f"应返回字符串，实际 {type(out)}"
    assert out.startswith("he"), f"应以 'he' 开头，实际 {out!r}"
    assert len(out) == 12, f"应是 2+10=12 个字符，实际 {len(out)}"


# ============================================================
print("=" * 60)
print("逐组件调试（按依赖顺序）")
print("=" * 60)

steps = [
    ("① PositionalEncoding", test_pos_enc),
    ("② MultiHeadAttention", test_mha),
    ("③ FeedForward", test_ffn),
    ("④ TransformerBlock", test_block),
    ("⑤ MiniGPT.forward", test_model),
    ("⑥ train_model（2 步）", test_training_step),
    ("⑦ generate（10 个字符）", test_generate),
]

passed = 0
for name, fn in steps:
    ok = run(name, fn)
    if not ok:
        print()
        print(f"→ 先修这个，再往下测。后面的组件依赖它。")
        break
    passed += 1

print()
print(f"通过 {passed} / {len(steps)}")
if passed == len(steps):
    print("全部通过 ✅ 可以跑完整的 lesson11.py 了")
