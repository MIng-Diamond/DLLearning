"""
项目一 · 第 1 节 —— 检查
==========================
    cd ~/projects/learn/DeepLearning/03_housing
    python check1.py
"""

import numpy as np
from data import (
    load_data, split_and_standardize,
    FEATURE_NAMES, TARGET_NAME,
)

N_TOTAL = 20640
N_TEST = 4128
N_TRAIN = 16512


def check_load():
    df = load_data()

    assert df.shape == (N_TOTAL, 9), \
        f"DataFrame 应为 (20640, 9)，实际 {df.shape}"

    for name in FEATURE_NAMES + [TARGET_NAME]:
        assert name in df.columns, f"缺少列 {name}，实际列名：{list(df.columns)}"

    # 数值合理性：收入中位数不可能是负数，房价应该是正数
    assert df["MedInc"].min() >= 0, f"MedInc 不应为负，最小 {df['MedInc'].min()}"
    assert df[TARGET_NAME].min() > 0, f"房价应为正数，最小 {df[TARGET_NAME].min()}"

    print(f"    {df.shape[0]} 个街区 × {df.shape[1]} 列")
    print(f"    特征量纲差异: MedInc {df['MedInc'].min():.1f}~{df['MedInc'].max():.1f}"
          f"   vs   Population {df['Population'].min():.0f}~{df['Population'].max():.0f}")


def check_split():
    df = load_data()
    X_tr, y_tr, X_te, y_te, mean, std = split_and_standardize(df)

    # —— 形状 ——
    assert X_tr.shape == (N_TRAIN, 8), f"X_train 应为 (16512, 8)，实际 {X_tr.shape}"
    assert X_te.shape == (N_TEST, 8), f"X_test 应为 (4128, 8)，实际 {X_te.shape}"
    assert y_tr.shape == (N_TRAIN,), f"y_train 应为 (16512,)，实际 {y_tr.shape}"
    assert y_te.shape == (N_TEST,), f"y_test 应为 (4128,)，实际 {y_te.shape}"
    assert mean.shape == (8,), f"mean 应为 (8,)，实际 {mean.shape}"
    assert std.shape == (8,), f"std 应为 (8,)，实际 {std.shape}"
    assert np.all(std > 0), f"标准差必须都为正，实际 {std}"

    # —— 训练集必须被【精确】标准化 ——
    # 只有「用训练集自己的统计量」才能得到精确的 0 和 1。
    # 如果你误用了全体数据的统计量，这里就会差出 1e-2 量级，检查会抓到。
    train_mean_err = np.abs(X_tr.mean(axis=0)).max()
    train_std_err = np.abs(X_tr.std(axis=0) - 1).max()
    assert train_mean_err < 1e-10, \
        f"训练集均值应为 0，实际最大偏差 {train_mean_err:.2e} " \
        f"—— 你是不是用了全体数据（或测试集）的统计量？那就是数据泄漏"
    assert train_std_err < 1e-10, \
        f"训练集标准差应为 1，实际最大偏差 {train_std_err:.2e}"

    # —— 测试集：反标准化后应该回到原始量纲 ——
    # 这验证了 mean/std 确实是对应的那些统计量
    X_te_raw = X_te * std + mean
    assert X_te_raw[:, 0].max() < 20, \
        f"反标准化后 MedInc 最大值 {X_te_raw[:, 0].max():.1f}，超出合理范围（原始 0~15）"
    assert X_te_raw[:, 4].max() > 1000, \
        f"反标准化后 Population 最大值 {X_te_raw[:, 4].max():.0f}，太小了（原始上万）"

    # —— 测试集不应被精确标准化（否则就是用了它自己的统计量）——
    test_mean_err = np.abs(X_te.mean(axis=0)).max()
    assert test_mean_err > 1e-8, \
        "测试集被精确标准化了 —— 说明你用了测试集自己的 mean/std，这是数据泄漏"

    # —— 打乱检查：两边的目标值分布应该接近 ——
    gap = abs(y_tr.mean() - y_te.mean())
    assert gap < 0.15, \
        f"训练集和测试集的房价均值差了 {gap:.3f} —— 划分可能没打乱"

    print(f"    训练集 {X_tr.shape}  测试集 {X_te.shape}")
    print(f"    训练集标准化误差: 均值 {train_mean_err:.1e}, 标准差 {train_std_err:.1e}  ✅")
    print(f"    测试集均值偏差 {test_mean_err:.4f}（不是 0，正确 —— 没有数据泄漏）")
    print(f"    mean 前 3 个: {np.round(mean[:3], 3)}")
    print(f"    std  前 3 个: {np.round(std[:3], 3)}")


def main():
    checks = [
        ("① 加载数据 load_data", check_load),
        ("② 划分 + 标准化 split_and_standardize", check_split),
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
        print("\n第 1 节完成 ✅  数据准备就绪，可以进第 2 节：训练 + 评估")


if __name__ == "__main__":
    main()
