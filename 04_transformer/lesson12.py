"""
第 12 课：Encoder-Decoder Transformer —— 序列反转
====================================================

【任务】

    输入:  [3, 7, 1, 9, 2]
    输出:  [2, 9, 1, 7, 3]

【为什么要做这个任务】

    1. 输入和输出都是序列 —— 这是 seq2seq 的最小版本
    2. 答案必须"先看完整段输入"才能给出 —— 单向 causal 模型做不到
    3. 不依赖语言数据，纯符号，对错一目了然

    学完这一课，你就有了机器翻译 / 语音识别 / 机器人指令跟随的骨架。

═══════════════════════════════════════════════════════════════
 架构
═══════════════════════════════════════════════════════════════

    输入 x ──► Embedding + 位置编码 ──► Encoder × N（双向自注意力）
                                              │
                                              ▼
                                          memory (B, L, D)
                                              │
    目标 y ──► Embedding + 位置编码 ──► Decoder × N
                                        （因果自注意力 + 交叉注意力）
                                              │
                                              ▼
                                        logits (B, L, V)

【Encoder 和 Decoder 的区别】

    Encoder: 双向自注意力（attn_mask=None）
             —— 输入整段可见，每个位置可以看所有位置

    Decoder: ① 因果自注意力（causal mask）
                —— 生成第 t 个词时，不能看第 t 个词之后的答案
             ② 交叉注意力（cross-attention）
                —— Q 来自 Decoder，K/V 来自 Encoder

═══════════════════════════════════════════════════════════════
 新东西：交叉注意力 Cross-Attention
═══════════════════════════════════════════════════════════════

    自注意力:   Q、K、V 都来自同一个序列
    交叉注意力: Q 来自 Decoder，K、V 来自 Encoder 的 memory

    一句话：
        Decoder 问："我下一步该输出什么？"
        Encoder 答："看我这里存着的输入信息。"

    形状：
        x      (B, T_dec, D)      ← Decoder 当前状态
        memory (B, T_enc, D)      ← Encoder 输出

        q = W_q(x)          (B, T_dec, D)
        k = W_k(memory)     (B, T_enc, D)
        v = W_v(memory)     (B, T_enc, D)

        SDPA(q, k, v) → (B, T_dec, D)
                         ↑ 输出长度跟着 q 走，不是 T_enc！

═══════════════════════════════════════════════════════════════
 两种 mask，别搞混
═══════════════════════════════════════════════════════════════

    Decoder 自注意力:  需要 causal mask（不能看未来的输出）
    交叉注意力:        不需要 mask（可以看到整个输入）
    Encoder 自注意力:  不需要 mask（输入整段可见）

═══════════════════════════════════════════════════════════════
 Teacher Forcing
═══════════════════════════════════════════════════════════════

    训练时，Decoder 的输入不是自己生成的，而是正确答案右移一位：

        目标 y:        [9, 1, 7, 3]
        Decoder 输入:  [BOS, 9, 1, 7]

    这样每个位置都能并行算损失，不用串行生成。
    推理时才一个词一个词地生成（greedy / 采样）。

    BOS = Beginning Of Sequence，编号取 vocab_size（比普通 token 大 1）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lesson11 import (
    PositionalEncoding, MultiHeadAttention, FeedForward,
    TransformerBlock, causal_mask,
)


# ============================================================
# 数据（已写好）
# ============================================================
def make_batch(batch_size, seq_len, vocab_size, device="cpu"):
    """随机造一批序列反转数据。

    x: (B, L)  元素是 0 ~ vocab_size-1 的整数
    y: (B, L)  把 x 沿序列维翻转
    """
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.flip(x, dims=[1])
    return x, y


def make_batch_fn(batch_size, seq_len, vocab_size, device="cpu"):
    """返回一个无参函数，每次调用都造一批新数据。"""
    def get_batch():
        return make_batch(batch_size, seq_len, vocab_size, device)
    return get_batch


# ============================================================
# ① 交叉注意力（新）
# ============================================================
class CrossAttention(nn.Module):
    """Q 来自 decoder，K/V 来自 encoder 的 memory。

    和自注意力的唯一区别：K、V 从 memory 投影，不是从 x。
    """

    def __init__(self, D, n_heads):
        super().__init__()
        assert D % n_heads == 0, "D 必须能被 n_heads 整除"
        self.D = D
        self.n_heads = n_heads
        self.hd = D // n_heads

        self.W_q = nn.Linear(D, D)
        self.W_k = nn.Linear(D, D)
        self.W_v = nn.Linear(D, D)
        self.W_o = nn.Linear(D, D)

    def forward(self, x, memory, attn_mask=None):
        """x: (B, T_dec, D)   memory: (B, T_enc, D) → (B, T_dec, D)

        TODO ①：照着 lesson11 的 MultiHeadAttention.forward 写，
                 只改三行：

            q = self.W_q(x)          # (B, T_dec, D)
            k = self.W_k(memory)     # (B, T_enc, D)   ← 不是 x
            v = self.W_v(memory)     # (B, T_enc, D)   ← 不是 x

        然后：
            拆头 → SDPA → 拼回 → self.W_o

        注意拆头时 q 用 T_dec，k/v 用 T_enc。
        """
        B, T_dec, _ = x.shape
        _, T_enc, _ = memory.shape
        q = self.W_q(x)
        k = self.W_k(memory)
        v = self.W_v(memory)
        q = q.reshape(B, T_dec, self.n_heads, self.hd).transpose(1, 2)
        k = k.reshape(B, T_enc, self.n_heads, self.hd).transpose(1, 2)
        v = v.reshape(B, T_enc, self.n_heads, self.hd).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1,2).reshape(B, T_dec, -1)
        return self.W_o(out)
        raise NotImplementedError("还没写 CrossAttention.forward")


# ============================================================
# ② Decoder Block
# ============================================================
class DecoderBlock(nn.Module):
    """Pre-LN，三个子层：

        x = x + SelfAttn(LN(x), causal_mask)     ← 因果自注意力
        x = x + CrossAttn(LN(x), memory)         ← 交叉注意力（不 mask）
        x = x + FFN(LN(x))
    """

    def __init__(self, D, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.self_attn = MultiHeadAttention(D, n_heads)
        self.ln2 = nn.LayerNorm(D)
        self.cross_attn = CrossAttention(D, n_heads)
        self.ln3 = nn.LayerNorm(D)
        self.ffn = FeedForward(D)

    def forward(self, x, memory, self_mask=None):
        """x: (B, T_dec, D)   memory: (B, T_enc, D) → (B, T_dec, D)

        TODO ②：三个残差块，和 lesson11 的 TransformerBlock 一个套路：

            x = x + self.self_attn(self.ln1(x), self_mask)
            x = x + self.cross_attn(self.ln2(x), memory)
            x = x + self.ffn(self.ln3(x))
            return x
        """
        x = x + self.self_attn(self.ln1(x), self_mask)
        x = x + self.cross_attn(self.ln2(x), memory)
        x = x + self.ffn(self.ln3(x))
        return x
        raise NotImplementedError("还没写 DecoderBlock.forward")


# ============================================================
# ③ 完整模型
# ============================================================
class EncoderDecoder(nn.Module):
    """Encoder（双向）+ Decoder（因果自注意力 + 交叉注意力）。"""

    def __init__(self, vocab_size, D=64, n_heads=4, n_enc=2, n_dec=2,
                 max_len=32):
        super().__init__()
        self.enc_emb = nn.Embedding(vocab_size, D)
        self.enc_pos = PositionalEncoding(D, max_len)
        self.encoder = nn.ModuleList([
            TransformerBlock(D, n_heads) for _ in range(n_enc)
        ])

        self.dec_emb = nn.Embedding(vocab_size, D)
        self.dec_pos = PositionalEncoding(D, max_len)
        self.decoder = nn.ModuleList([
            DecoderBlock(D, n_heads) for _ in range(n_dec)
        ])

        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab_size)

    def encode(self, x):
        """x: (B, L) → memory: (B, L, D)

        TODO ③：
            h = self.enc_pos(self.enc_emb(x))
            for block in self.encoder:
                h = block(h, attn_mask=None)     # 双向，不 mask
            return h
        """
        h = self.enc_emb(x)
        h = self.enc_pos(h)
        for block in self.encoder:
            h = block(h, attn_mask=None)
        return h
        raise NotImplementedError("还没写 EncoderDecoder.encode")

    def decode(self, y_in, memory, self_mask=None):
        """y_in: (B, L_dec) → logits: (B, L_dec, V)

        TODO ④：
            h = self.dec_pos(self.dec_emb(y_in))
            for block in self.decoder:
                h = block(h, memory, self_mask)
            return self.head(self.ln_f(h))
        """
        h = self.dec_emb(y_in)
        h = self.dec_pos(h)
        for block in self.decoder:
            h = block(h, memory, self_mask)
        return self.head(self.ln_f(h))
        raise NotImplementedError("还没写 EncoderDecoder.decode")

    def forward(self, x, y_in):
        """x: (B, L)  y_in: (B, L) → logits: (B, L, V)

        TODO ⑤：
            memory = self.encode(x)
            mask = causal_mask(y_in.size(1), x.device)
            return self.decode(y_in, memory, mask)
        """
        memory = self.encode(x)
        mask = causal_mask(y_in.shape[1], x.device)
        return self.decode(y_in, memory, mask)
        raise NotImplementedError("还没写 EncoderDecoder.forward")


# ============================================================
# ④ 训练（自己写）
# ============================================================
def train(model, get_batch, device, bos, steps=600, lr=3e-4, log_every=100):
    """训练循环。

    参数
        get_batch : 无参函数，返回 (x, y)，都是 (B, L)
        bos       : BOS 的编号（= 普通词表大小 vocab_size）

    每个 step：

        ① 取数据：
             x, y = get_batch()                     # (B, L)

        ② 右移构造 Decoder 输入：
             y_in = torch.cat([
                 torch.full((B, 1), bos, dtype=torch.long, device=device),
                 y[:, :-1],
             ], dim=1)                              # (B, L)

        ③ 前向：
             logits = model(x, y_in)                # (B, L, V+1)

        ④ 损失：
             loss = F.cross_entropy(
                 logits.reshape(-1, logits.size(-1)),   # (B*L, V+1)
                 y.reshape(-1),                        # (B*L,)
             )

        ⑤ 反向 + 梯度裁剪 + 更新

        ⑥ 每 log_every 步打印：
             loss 和 acc
             acc = (logits.argmax(-1) == y).float().mean()

    TODO ⑥：自己写完整循环，返回 losses（Python float 列表）。
    """
    losses = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for step in range(steps):
        x, y = get_batch()
        B, _ = x.shape
        y_in = torch.cat([torch.full((B, 1), bos, dtype=torch.long, device=device),
                          y[:, :-1]], dim=1)
        logits = model(x, y_in)
        optimizer.zero_grad()
        loss = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1)%log_every == 0:
            print(f"step: {step}  loss: {loss.item()}\n")
    return losses
    raise NotImplementedError("还没写 train")


# ============================================================
# ⑤ 推理：贪心解码
# ============================================================
@torch.no_grad()
def greedy_decode(model, x, bos):
    """x: (B, L) → y_hat: (B, L)  一次生成一个 token。

    TODO ⑦：
        ① memory = model.encode(x)                  # (B, L, D)
        ② ys = torch.full((B, 1), bos, ...)          # (B, 1)
        ③ 循环 L 次：
             mask = causal_mask(ys.size(1), x.device)
             logits = model.decode(ys, memory, mask)  # (B, T, V)
             next_tok = logits[:, -1, :].argmax(-1, keepdim=True)  # (B,1)
             ys = torch.cat([ys, next_tok], dim=1)
        ④ return ys[:, 1:]                           # 去掉 BOS
    """
    memory = model.encode(x)
    ys = torch.full((x.shape[0], 1), bos, dtype=torch.long, device=x.device)
    for _ in range(x.shape[1]):
        mask = causal_mask(ys.shape[1], x.device)
        logits = model.decode(ys, memory, mask)
        next_tok = logits[:, -1, :].argmax(-1, keepdim=True)
        ys = torch.cat([ys, next_tok], dim=1)
    return ys[:, 1:]
    raise NotImplementedError("还没写 greedy_decode")


# ============================================================
# 检查
# ============================================================
def check_cross_attention():
    torch.manual_seed(0)
    B, Td, Te, D, h = 2, 4, 6, 16, 4
    x = torch.randn(B, Td, D)
    memory = torch.randn(B, Te, D)
    ca = CrossAttention(D, h)
    out = ca(x, memory)
    assert out.shape == (B, Td, D), f"应为 {(B,Td,D)}，实际 {tuple(out.shape)}"
    assert out.shape[1] == Td, "输出长度应跟 query 走（T_dec），不是 T_enc"
    print(f"    输入 x {tuple(x.shape)} + memory {tuple(memory.shape)} → {tuple(out.shape)}  ✅")


def check_decoder_block():
    torch.manual_seed(0)
    B, Td, Te, D = 2, 5, 7, 16
    block = DecoderBlock(D, 4)
    x = torch.randn(B, Td, D)
    memory = torch.randn(B, Te, D)
    mask = causal_mask(Td, "cpu")
    out = block(x, memory, mask)
    assert out.shape == (B, Td, D), f"应为 {(B,Td,D)}，实际 {tuple(out.shape)}"
    print(f"    DecoderBlock: {tuple(x.shape)} + memory {tuple(memory.shape)} → {tuple(out.shape)}  ✅")


def check_forward():
    torch.manual_seed(0)
    V, B, L, D = 10, 2, 6, 32
    model = EncoderDecoder(V + 1, D=D, n_heads=4, n_enc=1, n_dec=1, max_len=16)
    x, y = make_batch(B, L, V)
    y_in = torch.cat([torch.full((B, 1), V), y[:, :-1]], dim=1)
    out = model(x, y_in)
    assert out.shape == (B, L, V + 1), f"应为 {(B,L,V+1)}，实际 {tuple(out.shape)}"
    print(f"    x {tuple(x.shape)} + y_in {tuple(y_in.shape)} → logits {tuple(out.shape)}  ✅")


def check_overfit():
    """在 16 条固定样本上过拟合，准确率必须到 100%。"""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    V, L, B = 8, 6, 16
    x, y = make_batch(B, L, V, device)
    model = EncoderDecoder(V + 1, D=64, n_heads=4, n_enc=2, n_dec=2,
                           max_len=16).to(device)

    train(model, lambda: (x, y), device, bos=V, steps=300, log_every=150)

    model.eval()
    y_in = torch.cat([torch.full((B, 1), V, device=device), y[:, :-1]], dim=1)
    with torch.no_grad():
        logits = model(x, y_in)
    acc = (logits.argmax(-1) == y).float().mean().item()
    assert acc == 1.0, f"过拟合固定样本应 100%，实际 {acc:.1%}"
    print(f"    16 条固定样本过拟合准确率 {acc:.1%}  ✅")


def check_generalize():
    """随机数据训练，再在没见过的序列上测准确率。"""
    torch.manual_seed(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    V, L = 10, 8
    get_batch = make_batch_fn(64, L, V, device)
    model = EncoderDecoder(V + 1, D=64, n_heads=4, n_enc=2, n_dec=2,
                           max_len=16).to(device)

    train(model, get_batch, device, bos=V, steps=600, log_every=200)

    model.eval()
    x_test, y_test = make_batch(256, L, V, device)
    y_hat = greedy_decode(model, x_test, V)
    acc = (y_hat == y_test).float().mean().item()
    print(f"泛化准确率应 ≥ 95%，实际 {acc:.1%}\n")
    assert acc >= 0.95, f"泛化准确率应 ≥ 95%，实际 {acc:.1%}"

    print(f"    没见过的 256 条序列，整句全对比例 {acc:.1%}  ✅")
    print("    例子:")
    for i in range(3):
        print(f"      输入 {x_test[i].tolist()} → 预测 {y_hat[i].tolist()} "
              f"(正确 {y_test[i].tolist()})")


def main():
    checks = [
        ("① CrossAttention", check_cross_attention),
        ("② DecoderBlock", check_decoder_block),
        ("③ EncoderDecoder.forward", check_forward),
        ("④ 过拟合 16 条固定样本", check_overfit),
        ("⑤ 泛化到新序列（≥95%）", check_generalize),
    ]

    passed = 0
    for name, fn in checks:
        try:
            fn()
            print(f"[PASS] {name}")
            passed += 1
        except NotImplementedError as e:
            print(f"[SKIP] {name} —— {e}")
            break
        except AssertionError as e:
            print(f"[FAIL] {name} —— {e}")
            break
        except Exception as e:
            tb = e.__traceback__
            while tb.tb_next:
                tb = tb.tb_next
            print(f"[ERROR] {name} —— 第 {tb.tb_lineno} 行: {type(e).__name__}: {e}")
            break

    print(f"\n通过 {passed} / {len(checks)}")
    if passed == len(checks):
        print("第 12 课完成 ✅  阶段 3 过关 —— 你已经会写 Encoder-Decoder Transformer")


if __name__ == "__main__":
    main()
