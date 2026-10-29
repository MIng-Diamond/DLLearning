"""
第 13 课：在真实语料上训练 Mini-GPT（tiny Shakespeare）
=========================================================

lesson11 的 MiniGPT 只在 400 个字符上训练，本质是"背诵"。
这一课换成约 110 万字符的莎士比亚全集，看它能不能学到"像英文"的生成。

【和 lesson11 的区别】

    语料      400 字符          → 约 1,115,394 字符
    词表      40                → 约 65
    D         128               → 256
    层数      3                 → 6
    block     64                → 128
    训练步数  2000              → 6000

【本课新增】

    ① 训练集 / 验证集划分（验证集不参与训练，用来判断是否过拟合）
    ② 学习率 warmup + cosine（大模型训练的标配）
    ③ 验证集 loss / perplexity
    ④ checkpoint 保存
    ⑤ temperature 对比实验

【本课用到的 API】

    torch.save(obj, path)
        作用：把 Python 对象（这里是 dict）序列化到磁盘
        返回：None
        坑：保存 state_dict（张量），不要直接保存整个 model 对象

    torch.load(path, map_location=...)
        作用：读回来
        返回：保存时的那个对象
        坑：GPU 上保存的，CPU 上读要加 map_location="cpu"

    math.exp(x)
        作用：e 的 x 次方
        用途：perplexity = exp(loss)

【运行】

    # 先快速跑 300 步，确认管线没问题（1~3 分钟）
    python lesson13_shakespeare.py --quick

    # 正式训练 6000 步
    python lesson13_shakespeare.py

【TODO（留给你）】

    TODO ①：实现 perplexity(loss)
    TODO ②：跑完后对比不同 temperature 的生成结果，把观察写进报告
"""

import math
import os
import sys
import time

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lesson11 import MiniGPT, generate, causal_mask

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "shakespeare.txt")
CKPT_PATH = os.path.join(HERE, "minigpt_shakespeare.pt")
PLOT_PATH = os.path.join(HERE, "lesson13_result.png")


# ============================================================
# 数据
# ============================================================
def load_text(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到 {path}\n"
            "先下载语料：\n"
            "  mkdir -p 04_transformer/data\n"
            "  wget -O 04_transformer/data/shakespeare.txt \\\n"
            "    https://cdn.jsdelivr.net/gh/karpathy/char-rnn@master/"
            "data/tinyshakespeare/input.txt"
        )
    if os.path.getsize(path) == 0:
        raise ValueError(f"{path} 是空文件，重新下载")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def encode(text, stoi):
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def split_data(data, val_ratio=0.1):
    n = int(len(data) * (1 - val_ratio))
    return data[:n], data[n:]


def make_get_batch(data, block_size, batch_size, device):
    """返回一个无参函数，每次随机切 batch_size 段长度为 block_size 的窗口。"""
    def get_batch():
        starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[s:s + block_size] for s in starts])
        y = torch.stack([data[s + 1:s + block_size + 1] for s in starts])
        return x.to(device), y.to(device)
    return get_batch


# ============================================================
# 学习率调度：warmup + cosine
# ============================================================
def get_lr(step, total_steps, base_lr, warmup_steps):
    """前 warmup_steps 步线性升到 base_lr，之后余弦降到 0.1 * base_lr。"""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (0.1 + 0.9 * cosine)


# ============================================================
# 评估
# ============================================================
@torch.no_grad()
def estimate_loss(model, get_batch, device, eval_iters=20):
    """在 eval_iters 个 batch 上算平均 loss。"""
    model.eval()
    total = 0.0
    for _ in range(eval_iters):
        x, y = get_batch()
        logits = model(x, causal_mask(x.size(1), device))
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total += loss.item()
    model.train()
    return total / eval_iters


def perplexity(loss):
    """困惑度：模型平均在多少个 token 之间犹豫。

    TODO ①：perplexity = exp(loss)

    提示：math.exp(loss)

    例子：loss = 1.5 → perplexity ≈ 4.48
    """
    return math.exp(loss)

    raise NotImplementedError("TODO ①：用 math.exp(loss) 实现 perplexity")


# ============================================================
# 训练
# ============================================================
def train(model, train_batch_fn, val_batch_fn, device,
          steps=6000, base_lr=3e-4, warmup_steps=200,
          log_every=100, eval_every=500, eval_iters=20):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr,
        betas=(0.9, 0.95), weight_decay=0.01,
    )

    train_losses = []
    val_points = []
    t0 = time.time()

    for step in range(steps):
        # 更新学习率
        lr = get_lr(step, steps, base_lr, warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = train_batch_fn()
        logits = model(x, causal_mask(x.size(1), device))
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_losses.append(loss.item())

        if (step + 1) % log_every == 0:
            print(f"step {step + 1:5d} | train loss {loss.item():.4f} "
                  f"| lr {lr:.2e} | {time.time() - t0:.0f}s")

        if (step + 1) % eval_every == 0 or step == steps - 1:
            val_loss = estimate_loss(model, val_batch_fn, device, eval_iters)
            val_points.append((step + 1, val_loss))
            print(f"         └─ val loss {val_loss:.4f}")

    return train_losses, val_points


# ============================================================
# 保存与画图
# ============================================================
def save_checkpoint(model, stoi, itos, model_cfg, path=CKPT_PATH):
    torch.save({
        "model": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "config": model_cfg,
    }, path)
    print(f"checkpoint 已保存: {path}")


def plot_losses(train_losses, val_points, path=PLOT_PATH):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#fcfcfb")

    ax.plot(train_losses, color="#9ecae1", linewidth=0.8,
            label="train（每步）", alpha=0.8)

    # 训练 loss 的滑动平均，便于看清趋势
    window = max(1, len(train_losses) // 100)
    smooth = torch.tensor(train_losses).unfold(0, window, 1).mean(1)
    ax.plot(range(window - 1, len(train_losses)), smooth,
            color="#2a78d6", linewidth=1.8, label=f"train 平滑（窗口 {window}）")

    xs = [p[0] for p in val_points]
    ys = [p[1] for p in val_points]
    ax.plot(xs, ys, "o-", color="#d62728", linewidth=1.5,
            markersize=4, label="val")

    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Mini-GPT on tiny Shakespeare")
    ax.grid(color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    plt.tight_layout()
    plt.savefig(path, dpi=130, facecolor="#fcfcfb")
    print(f"损失曲线已保存: {path}")


# ============================================================
# 主流程
# ============================================================
def main():
    quick = "--quick" in sys.argv
    steps = 300 if quick else 6000

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}   模式: {'快速测试' if quick else '正式训练'}")

    # ---- 数据 ----
    text = load_text()
    print(f"语料: {len(text):,} 字符")
    stoi, itos = build_vocab(text)
    vocab_size = len(stoi)
    print(f"词表: {vocab_size} 个字符")

    data = encode(text, stoi)
    train_data, val_data = split_data(data, val_ratio=0.1)
    print(f"训练集: {len(train_data):,} tokens   验证集: {len(val_data):,} tokens")

    block_size = 128
    batch_size = 64
    train_batch_fn = make_get_batch(train_data, block_size, batch_size, device)
    val_batch_fn = make_get_batch(val_data, block_size, batch_size, device)

    # ---- 模型 ----
    D, n_heads, n_layers = 256, 8, 6
    model = MiniGPT(vocab_size, D=D, n_heads=n_heads,
                    n_layers=n_layers, max_len=block_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型: D={D}, heads={n_heads}, layers={n_layers}, "
          f"参数量={n_params:,}")

    # ---- 训练 ----
    train_losses, val_points = train(
        model, train_batch_fn, val_batch_fn, device,
        steps=steps,
        base_lr=3e-4,
        warmup_steps=min(200, steps // 10),
        log_every=50 if quick else 100,
        eval_every=150 if quick else 500,
    )

    # ---- 最终评估 ----
    train_loss = estimate_loss(model, train_batch_fn, device, eval_iters=20)
    val_loss = estimate_loss(model, val_batch_fn, device, eval_iters=20)
    print(f"\n最终 train loss: {train_loss:.4f}")
    print(f"最终 val loss  : {val_loss:.4f}")

    try:
        print(f"最终 val perplexity: {perplexity(val_loss):.2f}")
    except NotImplementedError as e:
        print(f"[SKIP] {e}")

    # ---- 生成对比（TODO ② 的分析素材）----
    print("\n" + "=" * 60)
    print("temperature 对比（同一开头）")
    print("=" * 60)
    for temp in (0.5, 0.8, 1.0):
        print(f"\n--- temperature = {temp} ---")
        print(generate(model, "ROMEO:", stoi, itos, device,
                       n_chars=300, temperature=temp, top_k=40))

    # ---- 保存 ----
    plot_losses(train_losses, val_points)
    save_checkpoint(model, stoi, itos, {
        "vocab_size": vocab_size,
        "D": D,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "max_len": block_size,
    })

    print("\n第 13 课完成 ✅")
    print("TODO ②：把三种 temperature 的生成结果贴进报告，写下你的观察。")


if __name__ == "__main__":
    main()
