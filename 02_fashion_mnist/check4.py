"""
项目二 · 第 1 节 —— 检查
==========================

    cd 02_fashion_mnist
    python check4.py
"""

import torch

from data import get_loaders
from cnn import FashionCNN
from train import train, get_device


def check_model():
    model = FashionCNN()
    x = torch.randn(8, 1, 28, 28)
    out = model(x)
    assert out.shape == (8, 10), f"输出应为 (8,10)，实际 {tuple(out.shape)}"

    out2 = model(torch.randn(3, 1, 28, 28))
    assert out2.shape == (3, 10), \
        f"换 batch=3 后输出应为 (3,10)，实际 {tuple(out2.shape)}"

    n = sum(p.numel() for p in model.parameters())
    assert n == 20490, f"参数量应为 20,490，实际 {n}"
    assert len(list(model.parameters())) == 6, \
        f"应有 6 个参数张量（conv1/conv2/fc 各 weight+bias），实际 {len(list(model.parameters()))}"

    raw_sums = out.sum(dim=1)
    assert not torch.allclose(raw_sums, torch.ones(8), atol=1e-3), \
        "输出看起来已经 softmax 过了 —— forward 里不该加 softmax"

    print("    输出: (8,1,28,28) → (8,10)  ✅")
    print(f"    参数量: {n:,}，参数张量 6 个")


def check_train():
    device = get_device()
    train_loader, test_loader = get_loaders(batch_size=64)
    model = FashionCNN().to(device)

    history = train(model, train_loader, test_loader,
                    epochs=3, lr=0.02, device=device, verbose=False)

    acc = history["test_acc"][-1]
    assert acc > 0.85, f"3 轮后测试准确率 {acc:.3f}，应 > 0.85"
    print(f"    3 轮后 test_acc={acc:.3f}  ✅")


def main():
    torch.manual_seed(0)
    checks = [
        ("① CNN 结构（形状 / 参数量）", check_model),
        ("② 短训练（3 轮验证能学习）", check_train),
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
            print(f"[ERROR] {name} —— 第 {tb.tb_lineno} 行: "
                  f"{type(e).__name__}: {e}")

    print(f"\n通过 {passed} / {len(checks)}")
    if passed == len(checks):
        print("\n第 1 节完成 ✅ 接下来跑 python cnn.py 做完整训练")


if __name__ == "__main__":
    main()
