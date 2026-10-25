"""
项目 · 第 2 节 —— 检查
========================

跑这个文件，验证 train.py 的 evaluate() 和 train()。

    cd ~/projects/learn/DeepLearning/02_fashion_mnist
    python check2.py
"""

import torch
from torch.utils.data import DataLoader, Subset

from data import get_loaders
from model import FashionMLP
from train import train, get_device


def check_overfit():
    """20 个样本，模型应该能背下来（训练准确率接近 100%）。"""
    device = get_device()

    # 用第 1 节写好的 get_loaders 拿数据集，再切出前 20 个样本
    train_loader, _ = get_loaders(batch_size=20)
    small_set = Subset(train_loader.dataset, list(range(20)))
    small_loader = DataLoader(small_set, batch_size=20, shuffle=True)

    model = FashionMLP(hidden=256).to(device)
    history = train(model, small_loader, small_loader,
                    epochs=300, lr=0.1, device=device, verbose=False)

    assert history["train_acc"][-1] >= 0.95, \
        f"20 个样本都没背下来（train_acc={history['train_acc'][-1]:.3f}），训练循环有问题"
    assert history["train_loss"][-1] < 0.2, \
        f"小样本训练 loss 不够小（{history['train_loss'][-1]:.3f}）"

    print(f"    20 个样本: loss {history['train_loss'][0]:.3f} → "
          f"{history['train_loss'][-1]:.4f}, acc={history['train_acc'][-1]:.3f}")


def check_full():
    """全量训练 3 轮，测试准确率应该明显超过随机猜（10%）。"""
    device = get_device()
    train_loader, test_loader = get_loaders(batch_size=64)

    model = FashionMLP(hidden=256).to(device)
    history = train(model, train_loader, test_loader,
                    epochs=3, lr=0.1, device=device, verbose=False)

    assert history["test_acc"][0] > 0.5, \
        f"第 1 轮测试准确率只有 {history['test_acc'][0]:.3f}，训练没有生效"
    assert history["test_acc"][-1] > 0.80, \
        f"3 轮后测试准确率 {history['test_acc'][-1]:.3f}，应 > 0.80"
    assert history["test_loss"][-1] < history["test_loss"][0], \
        "测试 loss 没有下降"

    print(f"    3 轮后: test_acc={history['test_acc'][-1]:.3f}, "
          f"test_loss={history['test_loss'][-1]:.4f}")


def main():
    torch.manual_seed(0)

    checks = [
        ("① 小样本过拟合（验证训练循环）", check_overfit),
        ("② 全量训练 3 轮（验证真的在学）", check_full),
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
        print("\n第 2 节完成 ✅ 训练循环跑通了")


if __name__ == "__main__":
    main()
