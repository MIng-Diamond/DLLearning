"""
交互 demo ①：Mini-GPT 莎士比亚生成器
=====================================

用法：

    cd ~/projects/learn/DeepLearning/demo
    conda activate learndl

    # 交互模式
    python demo_text.py

    # 只生成一次（方便脚本调用）
    python demo_text.py --once "ROMEO:"

交互命令：

    直接输入开头文本        → 生成
    :temp 0.5              → 设置 temperature（默认 0.8）
    :len 400               → 设置生成长度（默认 200）
    :quit                  → 退出

模型的 checkpoint 来自 lesson13：
    04_transformer/minigpt_shakespeare.pt
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSFORMER_DIR = os.path.join(HERE, "..", "04_transformer")
sys.path.insert(0, TRANSFORMER_DIR)

from lesson11 import MiniGPT, causal_mask

CKPT_PATH = os.path.join(TRANSFORMER_DIR, "minigpt_shakespeare.pt")


@torch.no_grad()
def stream_generate(model, prompt, stoi, itos, device,
                    n_chars=200, temperature=0.8, top_k=40):
    """逐字符生成，边生成边打印。"""
    unknown = sorted({c for c in prompt if c not in stoi})
    if unknown:
        print(f"[警告] 这些字符不在词表里，已跳过: {unknown}")

    ids = [stoi[c] for c in prompt if c in stoi]
    if not ids:
        ids = [stoi["\n"]]

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    max_ctx = model.pos_enc.pe.size(1)

    print(prompt, end="", flush=True)

    for _ in range(n_chars):
        idx_cond = idx[:, -max_ctx:]
        mask = causal_mask(idx_cond.size(1), device)
        logits = model(idx_cond, mask)[:, -1, :]

        probs = torch.softmax(logits / temperature, dim=-1)
        k = min(top_k, probs.size(-1))
        topv, _ = torch.topk(probs, k, dim=-1)
        probs = probs.masked_fill(probs < topv[:, -1:], 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        next_id = torch.multinomial(probs, 1)
        ch = itos[next_id.item()]
        print(ch, end="", flush=True)

        idx = torch.cat([idx, next_id], dim=1)

    print()
    return "".join(itos[i] for i in idx[0].tolist())


def load_model(device):
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"找不到 {CKPT_PATH}\n"
            "先跑 04_transformer/lesson13_shakespeare.py 训练模型。"
        )

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = MiniGPT(**cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, ckpt["stoi"], ckpt["itos"], cfg


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stoi, itos, cfg = load_model(device)

    n_params = sum(p.numel() for p in model.parameters())
    print("=" * 60)
    print("Mini-GPT 莎士比亚生成器")
    print("=" * 60)
    print(f"设备: {device}   参数量: {n_params:,}")
    print(f"配置: D={cfg['D']}, layers={cfg['n_layers']}, "
          f"heads={cfg['n_heads']}, block={cfg['max_len']}")
    print()

    # --once 模式
    if "--once" in sys.argv:
        i = sys.argv.index("--once")
        prompt = sys.argv[i + 1] if i + 1 < len(sys.argv) else "ROMEO:"
        stream_generate(model, prompt, stoi, itos, device,
                        n_chars=300, temperature=0.8)
        return

    # 交互模式
    temperature = 0.8
    n_chars = 200

    print("直接输入开头文本就开始生成。")
    print("命令: :temp 0.5   :len 400   :quit")
    print("-" * 60)

    while True:
        try:
            line = input("\n提示> ")
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        line = line.strip()
        if not line:
            continue

        if line == ":quit":
            print("再见。")
            break
        if line.startswith(":temp"):
            try:
                temperature = float(line.split()[1])
                print(f"[temperature = {temperature}]")
            except (IndexError, ValueError):
                print("用法: :temp 0.8")
            continue
        if line.startswith(":len"):
            try:
                n_chars = int(line.split()[1])
                print(f"[生成长度 = {n_chars}]")
            except (IndexError, ValueError):
                print("用法: :len 400")
            continue

        stream_generate(model, line, stoi, itos, device,
                        n_chars=n_chars, temperature=temperature)


if __name__ == "__main__":
    main()
