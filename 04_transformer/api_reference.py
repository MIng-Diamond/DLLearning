"""
lesson11 用到的 PyTorch API 速查（可直接运行）
================================================

    cd 04_transformer
    python api_reference.py

每个 API 按统一格式讲：
    作用 / 签名 / 形状 / 返回什么 / 返回值的含义 / 例子 / 坑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def api(name, purpose, signature, shapes, returns, meaning, pitfalls):
    print("\n" + "=" * 66)
    print(f"{name}")
    print("=" * 66)
    print(f"  作用    : {purpose}")
    print(f"  签名    : {signature}")
    print(f"  形状    : {shapes}")
    print(f"  返回    : {returns}")
    print(f"  返回含义: {meaning}")
    if pitfalls:
        print(f"  坑      : {pitfalls}")


# ============================================================
api(
    "1. nn.Embedding —— 查表",
    "把整数索引映射成可学习的向量",
    "nn.Embedding(num_embeddings, embedding_dim)",
    "输入 (B, T) int64 → 输出 (B, T, D)",
    "一个张量 (B, T, D)",
    "每个索引对应的 D 维向量；内部是一张 (num_embeddings, D) 的可学习矩阵，"
    "第 i 行就是索引 i 的向量。训练时这些向量会被更新。",
    "输入必须是 int64；索引越界会 IndexError。",
)

emb = nn.Embedding(10, 4)
idx = torch.tensor([[1, 2, 3], [4, 5, 6]])
out = emb(idx)
print("  例子:")
print("    idx (B,T):", tuple(idx.shape), " dtype:", idx.dtype)
print("    out (B,T,D):", tuple(out.shape))
print("    索引 2 的向量 == weight[2]:", torch.allclose(out[0, 1], emb.weight[2]))


# ============================================================
api(
    "2. nn.Linear —— 全连接",
    "对最后一维做线性变换 y = xWᵀ + b",
    "nn.Linear(in_features, out_features)",
    "输入 (..., in) → 输出 (..., out)，前面的维度不变",
    "一个张量 (..., out)",
    "每个位置的向量都被同一个 W 和 b 变换；输出第 k 维是输入所有维的加权和。",
    "weight 的形状是 (out, in)，不是 (in, out)。",
)

lin = nn.Linear(8, 32)
x = torch.randn(2, 5, 8)
y = lin(x)
print("  例子:")
print("    (2,5,8) →", tuple(y.shape), " 前两维不动，最后一维 8→32")
print("    weight:", tuple(lin.weight.shape), " bias:", tuple(lin.bias.shape))


# ============================================================
api(
    "3. nn.LayerNorm —— 层归一化",
    "对每个样本的最后一维做标准化（减均值、除标准差）",
    "nn.LayerNorm(normalized_shape)",
    "输入和输出形状相同",
    "一个同形状张量",
    "每个位置自己的特征被归一化到均值 0、标准差 1，"
    "再乘可学习的 γ、加 β。不依赖 batch，训练/推理行为一致。",
    "normalized_shape 必须等于最后一维大小。",
)

ln = nn.LayerNorm(8)
x = torch.randn(2, 5, 8) * 10 + 3
y = ln(x)
print("  例子:")
print("    输入每行 mean/std:", round(float(x[0,0].mean()),2), round(float(x[0,0].std()),2))
print("    输出每行 mean/std:", round(float(y[0,0].detach().mean()),5),
      round(float(y[0,0].detach().std()),5),
      " ← std 不是精确 1，因为 torch.std 用无偏估计（除以 N-1）")


# ============================================================
api(
    "4. nn.GELU —— 激活函数",
    "非线性变换，负数平滑压到接近 0",
    "nn.GELU()",
    "输入输出形状相同",
    "一个同形状张量",
    "每个元素独立过 GELU；负数不会被直接截成 0，而是平滑衰减。",
    "Transformer 里比 ReLU 更常用。",
)

gelu = nn.GELU()
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print("  例子:")
print("    输入:", x.tolist())
print("    GELU:", [round(float(v), 3) for v in gelu(x)])
print("    ReLU:", [round(float(v), 3) for v in torch.relu(x)])


# ============================================================
api(
    "5. nn.Sequential —— 串联",
    "把多个层按顺序串成一个模块",
    "nn.Sequential(层1, 层2, ...)",
    "输入是层的输入，输出是最后一层的输出",
    "一个张量（最后一层的输出）",
    "数据依次流过每一层；等价于手写 x = layer1(x); x = layer2(x); ...",
    "有分支的结构（如残差）不能用 Sequential。",
)

net = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 8))
x = torch.randn(2, 5, 8)
print("  例子:")
print("    (2,5,8) →", tuple(net(x).shape))


# ============================================================
api(
    "6. nn.ModuleList —— 装层",
    "把多个层装进一个列表，并注册到模型",
    "nn.ModuleList([层1, 层2, ...])",
    "不是数据管道，是一个容器",
    "容器本身（用 for 循环取出每一层）",
    "和 Python list 的区别：ModuleList 里的层会被注册，参数会出现在 "
    "model.parameters() 里，也会跟着 .to(device) 搬。",
    "用普通 list 装层，参数不会注册，训练时不会更新！",
)

blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
x = torch.randn(2, 5, 8)
for b in blocks:
    x = b(x)
print("  例子:")
print("    3 个 Linear 串联后:", tuple(x.shape))
print("    ModuleList 参数量:", sum(p.numel() for p in blocks.parameters()))


# ============================================================
api(
    "7. F.scaled_dot_product_attention",
    "缩放点积注意力：softmax(q·kᵀ/√E)·v",
    "F.scaled_dot_product_attention(q, k, v, attn_mask=None, "
    "dropout_p=0.0, is_causal=False, scale=None)",
    "q (..., L, E)  k (..., S, E)  v (..., S, Ev) → (..., L, Ev)",
    "★ 一个张量 (..., L, Ev)。不是元组，也不返回注意力权重。",
    "对每个 query 位置 i，输出是【它关注的所有位置的 value 向量的加权平均】："
    "out[i] = Σ_j attn[i,j]·v[j]，权重由 q[i] 和 k[j] 的相似度决定。"
    "在 Transformer 里，它表示『该位置融合上下文之后的新表示』。",
    "bool mask 里 True = 允许关注；float mask 加到分数上，-inf = 禁止。"
    "别和 nn.MultiheadAttention 的 bool 语义搞混（那边 True = 屏蔽）。",
)

B, H, L, S, E = 2, 4, 5, 6, 8
q = torch.randn(B, H, L, E)
k = torch.randn(B, H, S, E)
v = torch.randn(B, H, S, E)
out = F.scaled_dot_product_attention(q, k, v)
print("  例子 1（形状）:")
print("    q/k/v:", tuple(q.shape), tuple(k.shape), tuple(v.shape))
print("    out  :", tuple(out.shape), "  ← 一个张量，不是 (out, attn)")

print("\n  例子 2（返回值的含义：value 的加权平均）:")
v3 = torch.tensor([[[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]])  # 红/绿/蓝
qq = torch.zeros(1, 1, 1, 3); kk = torch.zeros(1, 1, 3, 3)
qq[0,0,0] = torch.tensor([1., 0., 0.])
kk[0,0,2] = torch.tensor([10., 0., 0.])       # 只和位置 2 相似
o = F.scaled_dot_product_attention(qq, kk, v3)
print("    value: 位置0=红[1,0,0] 位置1=绿[0,1,0] 位置2=蓝[0,0,1]")
print("    out  :", [round(float(x),3) for x in o[0,0,0]], " ← 基本就是蓝：只看了位置 2")

qq2 = torch.zeros(1, 1, 1, 3); kk2 = torch.zeros(1, 1, 3, 3)
qq2[0,0,0] = torch.tensor([1., 0., 0.])
kk2[0,0,0] = torch.tensor([1., 0., 0.]); kk2[0,0,1] = torch.tensor([1., 0., 0.])
o2 = F.scaled_dot_product_attention(qq2, kk2, v3)
print("    两个 key 同样相似 → out:", [round(float(x),3) for x in o2[0,0,0]],
      " ← 红和绿的平均")

print("\n  例子 3（因果 mask）:")
q4 = k4 = v4 = torch.randn(1, 1, 4, 8)
out_c = F.scaled_dot_product_attention(q4, k4, v4, is_causal=True)
mask = torch.triu(torch.full((4, 4), float("-inf")), diagonal=1)
out_m = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=mask)
print("    is_causal=True == 手写上三角 -inf mask:",
      torch.allclose(out_c, out_m, atol=1e-6))


# ============================================================
api(
    "8. F.cross_entropy —— 交叉熵",
    "分类损失 = softmax + 负对数似然",
    "F.cross_entropy(input, target)",
    "input (N, C) 的 logits；target (N,) 的类别下标",
    "一个标量 loss",
    "每个样本正确类别的负对数概率的平均值。模型输出越接近正确类别，loss 越小。"
    "内部已经包含 softmax，所以别手动再加。",
    "input 必须是二维 (N, C)，所以要 reshape；target 必须是 int64。",
)

logits = torch.randn(2, 5, 10)
target = torch.randint(0, 10, (2, 5))
loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
print("  例子:")
print("    logits (B,T,V):", tuple(logits.shape), "→ reshape:", tuple(logits.reshape(-1,10).shape))
print("    target (B,T):", tuple(target.shape), "→ reshape:", tuple(target.reshape(-1).shape))
print("    loss:", round(float(loss), 4), " ← 一个标量")


# ============================================================
api(
    "9. torch.optim.AdamW",
    "优化器：根据梯度更新参数",
    "torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)",
    "输入模型参数，输出优化器对象",
    "一个优化器对象（不是张量）",
    "它保存每个参数的一阶/二阶矩，并在 step() 时更新参数。",
    "传 model.parameters()，不要传 model 本身。",
)

model = nn.Linear(8, 8)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss = model(torch.randn(4, 8)).pow(2).sum()
opt.zero_grad(); loss.backward(); opt.step()
print("  例子: zero_grad → backward → step 跑通")


# ============================================================
api(
    "10. clip_grad_norm_ —— 梯度裁剪",
    "把参数梯度的总范数限制在 max_norm 以内",
    "torch.nn.utils.clip_grad_norm_(parameters, max_norm)",
    "输入参数迭代器和阈值",
    "★ 一个标量：裁剪【前】的总范数",
    "返回的是裁剪前的梯度范数，方便你监控梯度是否爆炸。裁剪本身是原地修改 gradient。",
    "要在 backward() 之后、step() 之前调用。",
)

model = nn.Linear(8, 8)
loss = model(torch.randn(4, 8)).pow(2).sum() * 1000
loss.backward()
before = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
after = torch.norm(torch.stack([p.grad.norm() for p in model.parameters()])).item()
print("  例子:")
print("    返回的裁剪前总范数:", round(float(before), 2))
print("    裁剪后总范数:", round(after, 4), "（≤ 1.0）")


# ============================================================
api(
    "11. topk / masked_fill / multinomial —— 采样三件套",
    "top-k 过滤 + 按概率采样",
    "torch.topk(x, k) / tensor.masked_fill(cond, value) / torch.multinomial(probs, n)",
    "topk: (..., V) → (..., k)；multinomial: (N,) 或 (N,C) → (n,) 或 (N,n)",
    "topk 返回【两个张量】(values, indices)；masked_fill 返回同形状新张量；"
    "multinomial 返回采样到的【下标】",
    "topk 取前 k 大的值和位置；masked_fill 把不满足条件的置 0；"
    "multinomial 按概率分布抽一个下标。",
    "topk 返回两个值，别只接一个。multinomial 只接受 1D/2D、浮点、非负、每行和>0 的输入"
    "（不必先归一化，内部会归一化；但 logits 有负数，必须先 softmax）；"
    "replacement 默认 False（无放回），一次抽多个要传 True。",
)

probs = torch.tensor([[0.1, 0.5, 0.2, 0.15, 0.05]])
topv, topi = torch.topk(probs, 3, dim=-1)
filtered = probs.masked_fill(probs < topv[:, -1:], 0.0)
filtered = filtered / filtered.sum(dim=-1, keepdim=True)
sample = torch.multinomial(filtered, 1)
print("  例子:")
print("    topk(3): values =", [round(float(x),3) for x in topv[0]],
      " indices =", topi[0].tolist())
print("    过滤后:", [round(float(x),3) for x in filtered[0]],
      " 和为", round(float(filtered.sum()),3))
print("    采样下标:", sample.item())


# ============================================================
api(
    "12. register_buffer",
    "把一个不训练的张量注册到模型",
    "self.register_buffer(name, tensor)",
    "任意形状",
    "None（不是张量，只是注册动作）",
    "注册后：不在 parameters 里（不训练），但在 state_dict 里（会保存），"
    "也会跟着 model.to(device) 搬。适合位置编码、running mean 等固定张量。",
    "别用普通属性存固定张量，否则 .to(device) 时不会跟着搬。",
)

class Demo(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("pe", torch.zeros(1, 10, 8))

m = Demo()
print("  例子:")
print("    state_dict 里的 key:", list(m.state_dict().keys()))
print("    parameters 里的 key:", list(m.named_parameters()), " ← 空，说明不训练")


print("\n" + "=" * 66)
print("全部跑完 ✅")
print("=" * 66)
