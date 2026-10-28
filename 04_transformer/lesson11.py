"""
第 11 课：Transformer Block + Mini-GPT
========================================

你已经有了零件（第 10 课的多头注意力）。这一课把它们组装成整机。

═══════════════════════════════════════════════════════════════
 Transformer Block 的结构
═══════════════════════════════════════════════════════════════

    x ──→ LayerNorm ──→ Multi-Head Attention ──→ (+) ──→
     │                                            ↑
     └────────────────────────────────────────────┘  残差

    ──→ LayerNorm ──→ FeedForward ──→ (+) ──→ 输出
     │                                 ↑
     └─────────────────────────────────┘  残差

四个关键设计：

【1. 残差连接 x + f(x)】
    如果没有残差，梯度要穿过很多层，又会像 RNN 一样衰减。
    有了残差，梯度可以直接流过"加法"，一条高速公路。
    （和 LSTM 的细胞状态是同一个思想）

【2. LayerNorm】
    对每个样本的最后一维做归一化（减均值、除标准差）。
    和 BatchNorm 的区别：
        BatchNorm: 对 batch 里同一特征做归一化 → 依赖 batch size
        LayerNorm: 对单个样本的所有特征做归一化 → 不依赖 batch

    为什么 Transformer 用 LayerNorm？
        序列长度可变、batch 可能很小，BatchNorm 统计量不稳。

【3. FeedForward（FFN）】
    对每个位置独立地做：
        Linear(D, 4D) → GELU → Linear(4D, D)
    先升维再降维。作用是给模型更多非线性表达能力。

【4. Pre-LN 还是 Post-LN】
    原始论文是 Post-LN：  x = LayerNorm(x + f(x))
    现代实现多用 Pre-LN： h = LayerNorm(x); x = x + f(h)
    Pre-LN 训练更稳定，不需要 learning rate warmup。本课用 Pre-LN。

═══════════════════════════════════════════════════════════════
 位置编码（Positional Encoding）
═══════════════════════════════════════════════════════════════

Attention 有个"缺陷"：它对顺序不敏感。

    "狗咬人" 和 "人咬狗"，在 Attention 眼里是同一堆词。

所以要给每个位置加一个"位置标记"：

    PE(pos, 2i)   = sin(pos / 10000^(2i/D))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/D))

不同位置得到不同的正弦波组合。奇数维用 sin，偶数维用 cos。

═══════════════════════════════════════════════════════════════
 Mini-GPT：用 Transformer 做语言模型
═══════════════════════════════════════════════════════════════

    字符 → 嵌入 → +位置编码 → N 个 Transformer Block → Linear → 下一个字符的概率

训练任务：给前面的字符，预测下一个字符。
生成方式：一次预测一个字符，把它接到末尾，再预测下一个。
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


def _TODO(what):
    """占位符：这里要你自己填。运行到会明确提示还没写。"""
    raise NotImplementedError(f"还没写 {what}")


# ============================================================
# ① 位置编码
# ============================================================
class PositionalEncoding(nn.Module):
    """给每个位置加上独特的正弦波标记。"""

    def __init__(self, D, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, D)                     # (max_len, D)
        pos = torch.arange(max_len).unsqueeze(1).float() # (max_len, 1)

        # div_term: 10000^(2i/D)，用 exp/log 避免溢出
        div = torch.exp(torch.arange(0, D, 2).float() *
                        (-math.log(10000.0) / D))        # (D/2,)

        pe[:, 0::2] = torch.sin(pos * div)               # 偶数维用 sin
        pe[:, 1::2] = torch.cos(pos * div)               # 奇数维用 cos

        # register_buffer：作为模型的一部分保存，但不是可训练参数
        self.register_buffer("pe", pe.unsqueeze(0))      # (1, max_len, D)

    def forward(self, x):
        """x: (B, T, D) → (B, T, D)"""
        # TODO ①：把前 T 个位置编码加到 x 上
        # 提示：self.pe[:, :x.size(1)]
        return x + self.pe[:, : x.shape[1]]
        raise NotImplementedError("还没写 PositionalEncoding.forward")


# ============================================================
# ② 多头注意力模块（把第 10 课的函数封装成 nn.Module）
# ============================================================
class MultiHeadAttention(nn.Module):
    """手搭的多头注意力。

    和第 10 课的逻辑完全一样，只是封装成 Module：
        - 投影层放进 __init__
        - forward 里做 拆头 → SDPA → 拼回 → 输出投影
    """

    def __init__(self, D, n_heads):
        super().__init__()
        assert D % n_heads == 0, "D 必须能被 n_heads 整除"
        self.D = D
        self.n_heads = n_heads
        self.hd = D // n_heads

        # TODO ②：定义四个投影层
        #   self.W_q = nn.Linear(D, D)
        #   self.W_k = nn.Linear(D, D)
        #   self.W_v = nn.Linear(D, D)
        #   self.W_o = nn.Linear(D, D)
        self.W_q = nn.Linear(D, D)
        self.W_k = nn.Linear(D, D)
        self.W_v = nn.Linear(D, D)
        self.W_o = nn.Linear(D, D)
        #raise NotImplementedError("还没定义投影层")

    def forward(self, x, attn_mask=None):
        """x: (B, T, D) → (B, T, D)

        步骤：
            1. 投影：q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
            2. 拆头：reshape(B, T, h, hd).transpose(1, 2) → (B, h, T, hd)
            3. attention：
                 out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            4. 拼回：transpose(1,2).reshape(B, T, D)
            5. 输出投影：self.W_o(out)
            6. 返回 out
        """
        # TODO ③
        B, T, D = x.shape
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        q = q.reshape(B, T, self.n_heads, self.hd).transpose(1, 2) #B, h, T, hd
        k = k.reshape(B, T, self.n_heads, self.hd).transpose(1, 2) #B, h, T, hd
        v = v.reshape(B, T, self.n_heads, self.hd).transpose(1, 2) #B, h, T, hd

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)# B, h, T, hd
        out = out.transpose(1, 2).reshape(B, T, -1)
        out = self.W_o(out)
        return out
        raise NotImplementedError("还没写 MultiHeadAttention.forward")


# ============================================================
# ③ 前馈网络
# ============================================================
class FeedForward(nn.Module):
    """Linear(D, 4D) → GELU → Linear(4D, D)"""

    def __init__(self, D, mult=4):
        super().__init__()
        # TODO ④：定义 self.net = nn.Sequential(...)
        #   nn.Linear(D, D * mult), nn.GELU(), nn.Linear(D * mult, D)
        self.net = nn.Sequential(nn.Linear(D, mult * D),
                                 nn.GELU(),
                                 nn.Linear(mult * D, D))
        #raise NotImplementedError("还没定义 FeedForward")

    def forward(self, x):
        return self.net(x)


# ============================================================
# ④ Transformer Block
# ============================================================
class TransformerBlock(nn.Module):
    """Pre-LN 结构：

        x = x + Attention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))
    """

    def __init__(self, D, n_heads):
        super().__init__()
        # TODO ⑤：定义四个子层
        #   self.ln1  = nn.LayerNorm(D)
        #   self.attn = MultiHeadAttention(D, n_heads)
        #   self.ln2  = nn.LayerNorm(D)
        #   self.ffn  = FeedForward(D)
        self.ln1 = nn.LayerNorm(D)
        self.attn = MultiHeadAttention(D, n_heads)
        self.ln2 = nn.LayerNorm(D)
        self.ffn = FeedForward(D)
        #raise NotImplementedError("还没定义 TransformerBlock")

    def forward(self, x, attn_mask=None):
        # TODO ⑥：两个残差块
        #   h = self.ln1(x)
        #   x = x + self.attn(h, attn_mask)
        #   x = x + self.ffn(self.ln2(x))
        #   return x
        h = self.ln1(x)
        x = x + self.attn(h, attn_mask)
        x = x + self.ffn(self.ln2(x))
        return x 

        raise NotImplementedError("还没写 TransformerBlock.forward")


# ============================================================
# ⑤ Mini-GPT（已写好）
# ============================================================
class MiniGPT(nn.Module):
    """把零件组装成语言模型。

    数据流：
        idx (B,T) → 嵌入 → 位置编码 → N 个 Block → LayerNorm → Linear → logits
    """

    def __init__(self, vocab_size, D=128, n_heads=4, n_layers=2,
                 max_len=128):
        super().__init__()

        # TODO ⑦：嵌入层
        #   API：nn.Embedding(num_embeddings, embedding_dim)
        #     - num_embeddings = 词表大小 vocab_size
        #     - embedding_dim  = 向量维度 D
        #   作用：把整数索引 (B,T) 查表变成向量 (B,T,D)
        #   提示：内部是一张 (vocab_size, D) 的可学习矩阵
        self.tok_emb = nn.Embedding(vocab_size, D)#_TODO("⑦ 嵌入层 nn.Embedding")

        # TODO ⑧：位置编码（用上面自己写的 PositionalEncoding 类）
        self.pos_enc = PositionalEncoding(D, max_len)#_TODO("⑧ 位置编码")

        self.blocks = nn.ModuleList([
            TransformerBlock(D, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(D)

        # TODO ⑨：输出层
        #   API：nn.Linear(in_features, out_features)
        #     - 输入：每个位置的 D 维向量
        #     - 输出：每个位置上 vocab_size 个字符的分数（logits）

        self.head = nn.Linear(D, vocab_size)#_TODO("⑨ 输出层 nn.Linear")

    def forward(self, idx, attn_mask=None):
        """idx: (B, T) 的字符索引 → logits: (B, T, vocab_size)"""
        # TODO ⑩：查表 + 加位置编码
        #   x = self.tok_emb(idx)      # (B,T) → (B,T,D)
        #   x = self.pos_enc(x)        # (B,T,D) → (B,T,D)
        x = self.tok_emb(idx)#_TODO("⑩ 嵌入 + 位置编码")
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.ln_f(x)
        return self.head(x)


# ============================================================
# 数据与训练（已写好）
# ============================================================
TEXT = """
Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do: once or twice she had peeped into the book her
sister was reading, but it had no pictures or conversations in it, and what
is the use of a book, thought Alice, without pictures or conversations?

So she was considering in her own mind, as well as she could, for the hot day
made her feel very sleepy and stupid, whether the pleasure of making a
daisy-chain would be worth the trouble of getting up and picking the daisies,
when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very
much out of the way to hear the Rabbit say to itself, Oh dear! Oh dear! I
shall be late! But when the Rabbit actually took a watch out of its
waistcoat-pocket, and looked at it, and then hurried on, Alice started to her
feet, for it flashed across her mind that she had never before seen a rabbit
with either a waistcoat-pocket, or a watch to take out of it, and burning
with curiosity, she ran across the field after it, and fortunately was just
in time to see it pop down a large rabbit-hole under the hedge.
""".strip()


def build_dataset(text, block_size=64):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    def get_batch(batch_size = 64):
        starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[s:s + block_size] for s in starts])
        y = torch.stack([data[s + 1:s + block_size + 1] for s in starts])
        return x, y

    return get_batch, stoi, itos, len(chars)


def causal_mask(T, device):
    """上三角为 -inf，表示"不能看未来"。"""
    return torch.triu(torch.full((T, T), float("-inf"), device=device),
                      diagonal=1)


def train_model(model, get_batch, device, steps=2000, lr=1e-3, log_every=250):
    """训练 Mini-GPT。

    每一步做六件事：

        ① 取一批数据：
             xb, yb = get_batch(64)          # 都是 (B, T)

        ② 搬到设备：
             xb, yb = xb.to(device), yb.to(device)

        ③ 造因果 mask：
             mask = causal_mask(xb.size(1), device)

        ④ 前向：
             logits = model(xb, mask)        # (B, T, vocab_size)

        ⑤ 算交叉熵：
             loss = F.cross_entropy(
                 logits.reshape(-1, logits.size(-1)),   # (B*T, V)
                 yb.reshape(-1),                        # (B*T,)
             )
           为什么要 reshape？
               CrossEntropyLoss 要求输入是 (N, C)、标签是 (N,)。
               把 (B,T,V) 展平成 (B*T, V)，标签同理。

        ⑥ 反向 + 更新：
             optimizer.zero_grad()
             loss.backward()
             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
             # ↑ 梯度裁剪：把梯度范数限制在 1.0，防止 Transformer 训练初期爆炸
             optimizer.step()

    另外：
        - 优化器用 AdamW(model.parameters(), lr=lr)
        - 每一步把 loss.item() 存进列表
        - 每 log_every 步打印一次
        - 最后返回 losses
    """
    # TODO ⑪
    losses = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = F.cross_entropy
    for step in range(steps):
        x, y = get_batch()
        x, y = x.to(device), y.to(device)
        mask = causal_mask(x.shape[1], device)
        logits = model(x, mask)
        optimizer.zero_grad()
        loss = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1)%log_every == 0:
            print(f"step: {step}  loss: {loss.item()}\n")
    return losses
    raise NotImplementedError("还没写 train_model")


@torch.no_grad()
def generate(model, start_text, stoi, itos, device, n_chars=200,
             temperature=0.8, top_k=10):
    """自回归生成：一次一个字符。

    参数
        start_text  : 开头字符串，比如 "Alice"
        stoi / itos : 字符 ↔ 索引的映射
        temperature : 温度，越低越保守，越高越随机
        top_k       : 每一步只从概率最高的 k 个候选里采样

    步骤
        ① 把 start_text 编码成索引张量 (1, L)
        ② 循环 n_chars 次，每次：
             a. 滑动窗口：只取最近 max_ctx 个字符
                  max_ctx = model.pos_enc.pe.size(1)
             b. 造因果 mask
             c. 前向，只取【最后一个位置】的 logits      → (1, V)
             d. logits / temperature，再 softmax          → 概率
             e. top-k 过滤：只保留前 k 大的，其余置 0，再重新归一化
             f. 从概率分布里采样一个字符，接到 idx 末尾
        ③ 把索引序列解码成字符串

    常用 API             b. 造因果 mask
        torch.multinomial(probs, 1)        按概率采样一个
        torch.topk(probs, k, dim=-1)       取前 k 大
        probs.masked_fill(条件, 0.0)        把满足条件的置 0
        idx[:, -max_ctx:]                   最后 max_ctx 个字符

    ⚠️ 函数上方有 @torch.no_grad()，不用再手写 with torch.no_grad()。
    """
    # TODO ⑫
    idx = torch.tensor([[stoi[c] for c in start_text]], device=device)
    max_ctx = model.pos_enc.pe.shape[1]

    for _ in range(n_chars):
        ctx = idx[:, -max_ctx:]
        mask = causal_mask(ctx.shape[1], device=device)
        logits = model(ctx, mask)[:, -1, :]
        probs = torch.softmax(logits / temperature, dim = -1)
        k = min(top_k, probs.shape[-1])
        topv, _ = torch.topk(probs, k, dim=-1)
        probs = probs.masked_fill(probs < topv[:, -1:], 0.0)
        probs = probs / probs.sum(dim = -1, keepdim = True)
        next_id = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_id], dim = 1)
    return "".join(itos[i] for i in idx[0].tolist())
    raise NotImplementedError("还没写 generate")


# ============================================================
# 主流程
# ============================================================
def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    get_batch, stoi, itos, vocab_size = build_dataset(TEXT)
    print(f"词表大小: {vocab_size}")

    # 形状检查（不依赖 TODO 是否写完）
    try:
        model = MiniGPT(vocab_size, D=128, n_heads=4, n_layers=3).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {n_params:,}")
        xb, _ = get_batch(2)
        out = model(xb.to(device), causal_mask(xb.size(1), device))
        assert out.shape == (2, xb.size(1), vocab_size), \
            f"输出形状应为 {(2, xb.size(1), vocab_size)}，实际 {tuple(out.shape)}"
        print(f"[PASS] 前向形状: {tuple(xb.shape)} → {tuple(out.shape)}")
    except NotImplementedError as e:
        print(f"[SKIP] 前向形状 —— {e}")
        return
    except Exception as e:
        print(f"[FAIL] 前向形状 —— {type(e).__name__}: {e}")
        return

    print("\n开始训练 …")
    try:
        losses = train_model(model, get_batch, device, steps=2000)
    except NotImplementedError as e:
        print(f"[SKIP] 训练 —— {e}")
        return

    print("\n生成示例:")
    print("-" * 60)
    try:
        print(generate(model, "Alice", stoi, itos, device, n_chars=300))
    except NotImplementedError as e:
        print(f"[SKIP] 生成 —— {e}")
        return
    print("-" * 60)

    # 画损失曲线
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "lesson11_result.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#fcfcfb")
    ax.plot(losses, color="#2a78d6", linewidth=1.5)
    ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.set_title("Mini-GPT 训练损失")
    ax.grid(color="#e1e0d9", linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.savefig(path, dpi=120, facecolor="#fcfcfb")
    print(f"\n损失曲线已保存: {path}")
    print("\n第 11 课完成 ✅  你已经搭出了一个完整的 Transformer")


if __name__ == "__main__":
    main()
