"""
项目 · 第 1 节 —— 检查
========================
跑这个文件，验证 data.py 和 model.py 写对了。

    cd ~/projects/learn/DeepLearning/02_fashion_mnist
    python check1.py
"""

import sys
import torch

from data import get_loaders, CLASSES
from model import FashionMLP


def check_data():
    train_loader, test_loader = get_loaders(batch_size=64)

    # —— 数据集大小 ——
    assert len(train_loader.dataset) == 60000, \
        f"训练集应 60000 张，实际 {len(train_loader.dataset)}"
    assert len(test_loader.dataset) == 10000, \
        f"测试集应 10000 张，实际 {len(test_loader.dataset)}"

    # —— 一个 batch 的形状 ——
    xb, yb = next(iter(train_loader))
    assert xb.shape == (64, 1, 28, 28), \
        f"图像 batch 应为 (64,1,28,28)，实际 {tuple(xb.shape)}"
    assert yb.shape == (64,), f"标签 batch 应为 (64,)，实际 {tuple(yb.shape)}"
    assert xb.dtype == torch.float32, f"图像 dtype 应为 float32，实际 {xb.dtype}"

    # —— 预处理：像素要归一化到 0~1 ——
    assert 0.0 <= xb.min() and xb.max() <= 1.0, \
        f"像素范围 {xb.min():.3f}~{xb.max():.3f}，应该被 ToTensor 缩放到 0~1"

    # —— 标签范围 ——
    assert yb.min() >= 0 and yb.max() <= 9, f"标签应在 0~9，实际 {yb.min()}~{yb.max()}"

    # —— shuffle：训练集要打乱，测试集不能打乱 ——
    t1 = next(iter(train_loader))[1]
    t2 = next(iter(train_loader))[1]
    assert not torch.equal(t1, t2), \
        "训练集两次取到的顺序完全一样 —— shuffle=True 没写"

    s1 = next(iter(test_loader))[1]
    s2 = next(iter(test_loader))[1]
    assert torch.equal(s1, s2), \
        "测试集两次取到的顺序不一样 —— 测试集不该 shuffle"

    print(f"    训练集 60000 张 / 测试集 10000 张")
    print(f"    一个 batch: {tuple(xb.shape)}  像素 {xb.min():.2f}~{xb.max():.2f}")
    print(f"    类别数: {len(CLASSES)}")


def check_model():
    model = FashionMLP(hidden=256)

    # —— 前向能跑通，输出形状对 ——
    x = torch.randn(8, 1, 28, 28)
    out = model(x)
    assert out.shape == (8, 10), f"输出应为 (8,10)，实际 {tuple(out.shape)}"

    # —— 换 batch size 也要能跑（forward 不能写死 batch 大小）——
    out2 = model(torch.randn(3, 1, 28, 28))
    assert out2.shape == (3, 10), f"换 batch=3 后输出应为 (3,10)，实际 {tuple(out2.shape)}"

    # —— 参数量 ——
    n = sum(p.numel() for p in model.parameters())
    assert n == 203530, \
        f"参数量应为 203530（784×256+256 + 256×10+10），实际 {n}"
    assert len(list(model.parameters())) == 4, \
        f"应该有 4 个参数张量（fc1.weight, fc1.bias, fc2.weight, fc2.bias），实际 {len(list(model.parameters()))}"

    # —— 输出不该是 softmax 之后的（否则每行和为 1）——
    row_sums = out.softmax(dim=1).sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(8), atol=1e-5), "softmax 之后每行和应为 1"
    raw_sums = out.sum(dim=1)
    assert not torch.allclose(raw_sums, torch.ones(8), atol=1e-3), \
        "输出看起来已经 softmax 过了 —— forward 里不该加 softmax（CrossEntropyLoss 会做）"

    print(f"    参数量: {n:,}，参数张量 4 个")
    print(f"    输入 (8,1,28,28) → 输出 (8,10)  ✅")
    print(f"    换 batch=3 也能跑  ✅（说明 forward 没写死 batch 大小）")


def main():
    checks = [
        ("数据加载 data.py", check_data),
        ("模型 model.py", check_model),
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
        print("\n第 1 节完成 ✅  数据管线和模型都就绪了，可以进第 2 节：训练")


if __name__ == "__main__":
    main()
