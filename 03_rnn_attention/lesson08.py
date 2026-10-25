"""
第 08 课：RNN —— 让网络拥有「记忆」
=====================================

【为什么需要 RNN】

到目前为止，你的模型都是「看一眼，给一个答案」：
    CNN 看一张图 → 判断类别
    线性回归看一组特征 → 预测房价

但很多任务需要看【一串】数据，而且**顺序很重要**：
    语音识别：声音是一串波形
    机器人控制：过去几秒的状态 → 现在该往哪动
    语言：前面的词决定后面的词

「我今天不吃饭」vs「我今天不吃面」—— 顺序不同，意思不同。

【RNN 的核心：隐藏状态】

RNN 每一步都维护一个「隐藏状态」h，它相当于网络的记忆：

    h_t = tanh( x_t @ W_ih^T + b_ih + h_{t-1} @ W_hh^T + b_hh )

    读法：
        h_t      = 当前时刻的新记忆
        x_t      = 当前时刻的输入
        h_{t-1}  = 上一时刻的记忆   ← 这就是「记忆」的来源
        tanh     = 把值压到 -1~1

    每来一个新输入，就用「旧记忆 + 新输入」算出「新记忆」。

═══════════════════════════════════════════════════════════════
 本课要做的
═══════════════════════════════════════════════════════════════

1. 手写 RNN 前向，然后和 PyTorch 的 nn.RNN 逐位对照
2. 用 nn.LSTM 训练一个正弦波预测器：给 30 个点，预测第 31 个
"""

import numpy as np
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
# ① 手写 RNN 前向
# ============================================================
def rnn_forward_manual(x, rnn):
    """用 PyTorch 的 nn.RNN 里存着的权重，手写它的前向。

    x   : (B, T, D)      B 个序列，每个长 T，每步 D 维
    rnn : nn.RNN(input_size=D, hidden_size=H, batch_first=True)

    返回 : (B, T, H)     每一步的隐藏状态

    权重就在 rnn 上：
        rnn.weight_ih_l0 : (H, D)   输入 → 隐藏
        rnn.weight_hh_l0 : (H, H)   隐藏 → 隐藏
        rnn.bias_ih_l0   : (H,)
        rnn.bias_hh_l0   : (H,)

    ⚠️ PyTorch 的 Linear 用的是 x @ W.T，RNN 权重同理：
        x_t @ weight_ih_l0.T

    步骤：
        B, T, D = x.shape
        H = rnn.hidden_size
        h = torch.zeros(B, H)        # 初始记忆全 0
        outs = []
        for t in range(T):
            h = torch.tanh(x[:, t] @ Wih.T + bih + h @ Whh.T + bhh)
            outs.append(h)
        return torch.stack(outs, dim=1)   # (B, T, H)
    """
    # TODO ①
    B, T, D = x.shape
    x = x.permute(1, 0, 2) #x.shape (T, B, D)
    Wih = rnn.weight_ih_l0.T #(d, h)
    Whh = rnn.weight_hh_l0.T #(h, h)

    #Wih = Wih.unsqueeze(0).repeat(B, 1, 1)
    #Whh = Whh.unsqueeze(0).repeat(B, 1, 1)

    bih, bhh = rnn.bias_ih_l0, rnn.bias_hh_l0

    h = torch.zeros(B, rnn.hidden_size)
    outs = []
    for t in range(T):
        x_t = x[t]
        a = torch.mm(x_t, Wih)
        b = torch.mm(h, Whh)
        h = torch.tanh(a + b + bih + bhh)
        outs.append(h)
    return torch.stack(outs, dim = 1)
    raise NotImplementedError("还没写 rnn_forward_manual")


# ============================================================
# ② 正弦波数据
# ============================================================
def make_sine_data(n_windows=2000, seq_len=30, seed=0):
    """生成正弦波窗口，每个样本是「30 个点 → 第 31 个点」。

    返回 X: (N, 30, 1)   y: (N, 1)
    """
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(n_windows):
        start = rng.uniform(0, 20 * np.pi)
        t = np.arange(seq_len + 1) * 0.08
        wave = np.sin(start + t)
        xs.append(wave[:-1])
        ys.append(wave[-1])
    X = np.stack(xs).astype(np.float32)[:, :, None]   # (N, T, 1)
    y = np.array(ys, dtype=np.float32)[:, None]       # (N, 1)
    return X, y


# ============================================================
# ③ 模型：LSTM + 一个线性层
# ============================================================
class SineLSTM(nn.Module):
    """LSTM 读完整段序列，用最后一个时刻的隐藏状态预测下一个值。

    LSTM 是 RNN 的升级版，多了「门」来控制记住/忘掉哪些信息，
    能缓解 RNN 的梯度消失问题（第 09 课细讲）。
    """

    def __init__(self, hidden=64):
        super().__init__()
        # TODO ②：定义 LSTM 和输出层
        #   self.rnn = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        #   self.fc  = nn.Linear(hidden, 1)
        self.rnn = nn.LSTM(input_size = 1, hidden_size = hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
        #raise NotImplementedError("还没定义模型")

    def forward(self, x):
        """x: (B, T, 1) → (B, 1)

        步骤：
            out, _ = self.rnn(x)        # out: (B, T, H)
            取最后一个时间步 out[:, -1]  # (B, H)
            过 self.fc 得到 (B, 1)
        """
        # TODO ③
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])
        raise NotImplementedError("还没写 forward")


# ============================================================
# ④ 训练（已写好）
# ============================================================
def train_model(model, X, y, epochs=60, lr=1e-2, batch_size=64, device="cpu"):
    model.to(device)
    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor(y, device=device)
    n = len(X_t)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    losses = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            pred = model(X_t[idx])
            loss = criterion(pred, y_t[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        scheduler.step()
        losses.append(total / n)
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:2d}/{epochs}  loss={losses[-1]:.6f}")
    return losses


# ============================================================
# ⑤ 可视化（已写好）
# ============================================================
def plot_results(model, X, y, losses, device="cpu",
                 path="03_rnn_attention/lesson08_result.png"):
    import os
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.path.basename(path))
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, device=device)).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("#fcfcfb")

    axes[0].plot(losses, color="#2a78d6", linewidth=2)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("训练损失")
    axes[0].set_yscale("log")
    axes[0].grid(color="#e1e0d9", linewidth=0.8)
    axes[0].set_axisbelow(True)

    axes[1].scatter(y, pred, s=10, alpha=0.4, color="#2a78d6", edgecolors="none")
    lim = [min(y.min(), pred.min()) - 0.1, max(y.max(), pred.max()) + 0.1]
    axes[1].plot(lim, lim, "--", color="#898781", linewidth=1.2)
    axes[1].set_xlabel("真实值")
    axes[1].set_ylabel("预测值")
    axes[1].set_title("预测 vs 真实")
    axes[1].grid(color="#e1e0d9", linewidth=0.8)
    axes[1].set_axisbelow(True)

    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    plt.tight_layout()
    plt.savefig(path, dpi=120, facecolor="#fcfcfb")
    print(f"    图片已保存: {path}")


# ============================================================
# 检查
# ============================================================
def check_manual_rnn():
    torch.manual_seed(0)
    B, T, D, H = 3, 5, 4, 6

    rnn = nn.RNN(input_size=D, hidden_size=H, batch_first=True)
    x = torch.randn(B, T, D)

    mine = rnn_forward_manual(x, rnn)
    official, _ = rnn(x)

    assert mine.shape == (B, T, H), f"形状应为 {(B,T,H)}，实际 {tuple(mine.shape)}"
    assert torch.allclose(mine, official, atol=1e-6), \
        f"和 nn.RNN 不一致，最大差 {float((mine-official).abs().max()):.2e}"
    print(f"    (B,T,D)=({B},{T},{D}) → (B,T,H)={(B,T,H)}，与 nn.RNN 一致  ✅")


def check_model_and_train():
    torch.manual_seed(0)
    X, y = make_sine_data(n_windows=2000, seq_len=30)
    X_tr, y_tr = X[:1600], y[:1600]
    X_te, y_te = X[1600:], y[1600:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SineLSTM(hidden=64)
    losses = train_model(model, X_tr, y_tr, epochs=60, device=device)

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_te, device=device)).cpu().numpy()
    mse = float(np.mean((pred - y_te) ** 2))
    print(f"    测试 MSE = {mse:.6f}")
    assert mse < 0.01, f"测试 MSE {mse:.6f} 太大，应 < 0.01"

    return model, losses, X, y, device


def main():
    checks = [("① 手写 RNN 前向（对照 nn.RNN）", check_manual_rnn)]

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

    try:
        print()
        model, losses, X, y, device = check_model_and_train()
        print("[PASS] ②③ LSTM 训练正弦波预测")
        passed += 1
        plot_results(model, X, y, losses, device)
    except NotImplementedError as e:
        print(f"[SKIP] ②③ LSTM 训练 —— {e}")
    except AssertionError as e:
        print(f"[FAIL] ②③ LSTM 训练 —— {e}")
    except Exception as e:
        tb = e.__traceback__
        while tb.tb_next:
            tb = tb.tb_next
        print(f"[ERROR] ②③ LSTM 训练 —— 第 {tb.tb_lineno} 行: {type(e).__name__}: {e}")

    print(f"\n通过 {passed} / 2")
    if passed == 2:
        print("\n第 08 课完成 ✅  你理解了 RNN 的记忆机制，并用 LSTM 做了序列预测。")


if __name__ == "__main__":
    main()
