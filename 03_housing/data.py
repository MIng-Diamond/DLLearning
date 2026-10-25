"""
项目一 · 第 1 节 —— 数据
==========================

【项目目标】用线性回归预测加州某个街区的房价中位数。

【数据集】California Housing（1990 年加州人口普查）
  - 20640 个街区（block group）
  - 每个街区 8 个特征，1 个目标值
  - 这是一个【真实】数据集，不是人造的

【8 个特征】
    MedInc       街区收入中位数（万美元）
    HouseAge     房龄中位数
    AveRooms     平均房间数
    AveBedrms    平均卧室数
    Population   街区人口
    AveOccup     平均每户人数
    Latitude     纬度
    Longitude    经度

【目标】
    MedHouseVal  房价中位数（单位：10 万美元）

═══════════════════════════════════════════════════════════════
 本节要学的两个新东西（都是真实项目里的必备技能）
═══════════════════════════════════════════════════════════════

【一、特征标准化】

看上面 8 个特征，它们的量纲差得离谱：
    MedInc        0 ~ 15
    Population    3 ~ 35000
    Latitude      32 ~ 42

如果不处理，直接扔给梯度下降会出大问题 ——
不同特征的梯度量级差几万倍，学习率根本没法选：
调大到适合小特征，大特征就爆炸；调小到适合大特征，小特征就几乎不动。

标准化的做法：让每个特征都变成「均值 0、标准差 1」

    x_标准化 = (x - 均值) / 标准差

这样所有特征就站在同一起跑线上了。

【二、数据泄漏（data leakage）—— 这是本节的坑】

标准化要算均值和标准差。问题来了：**用谁的数据算？**

    ❌ 用全部数据（训练集 + 测试集）算  →  数据泄漏！
    ✅ 只用【训练集】算，然后套用到测试集

为什么？因为测试集代表「未来会遇到的新数据」。
你在训练时偷看了它的统计信息，评估结果就会虚高 ——
这相当于考试前偷看了考卷的难度分布。

**这条规矩在深度学习里同样适用，而且更严格。**
记住：凡是「从数据里算出来的东西」（均值、标准差、最大最小值、
词表、类别映射...），都只能从训练集算。
═══════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]
TARGET_NAME = "MedHouseVal"

# ============================================================
# ① 加载数据
# ============================================================
def load_data():
    """下载并返回 DataFrame（20640 行 × 9 列：8 个特征 + 1 个目标）。

    提示：
        data = fetch_california_housing(as_frame=True)
        return data.frame

    as_frame=True 让它直接返回 pandas DataFrame（而不是裸 numpy 数组），
    列名已经起好了，方便我们按名字取列。

    ⚠️ 第一次运行会联网下载（约 1MB），之后就缓存在本地了。
    """
    # TODO ①
    data = fetch_california_housing(as_frame=True)
    return data.frame
    raise NotImplementedError("还没写 load_data")


# ============================================================
# ② 划分 + 标准化（本节重点）
# ============================================================
def split_and_standardize(df, test_ratio=0.2, seed=0):
    """把数据划成训练集/测试集，并做标准化。

    返回 (X_train, y_train, X_test, y_test, mean, std)

    ── 步骤 ──

    1. 取出特征矩阵 X 和目标 y：
           X = df[FEATURE_NAMES].to_numpy()      # (20640, 8)
           y = df[TARGET_NAME].to_numpy()        # (20640,)

    2. 打乱索引并划分：
           rng = np.random.default_rng(seed)
           idx = rng.permutation(len(df))        # 打乱后的索引
           n_test = int(len(df) * test_ratio)
           test_idx  = idx[:n_test]
           train_idx = idx[n_test:]

       ⚠️ 为什么要打乱？数据可能按地理位置排序，不打乱的话
          训练集全是南加州、测试集全是北加州，评估就没意义了。

    3. 【关键】标准化用的 mean/std 只能从【训练集】算：
           mean = X_train_raw.mean(axis=0)       # (8,)
           std  = X_train_raw.std(axis=0)        # (8,)
           X_train = (X_train_raw - mean) / std
           X_test  = (X_test_raw  - mean) / std   ← 注意：用的是【训练集】的 mean/std！

    4. 返回 6 个东西（后面画图要用到 mean/std 来反标准化）

    ── 形状提醒 ──
       20640 × 0.2 = 4128 个测试样本，剩下 16512 个训练样本
       X_train (16512, 8)   y_train (16512,)
       X_test  ( 4128, 8)   y_test  ( 4128,)
       mean (8,)            std (8,)
    """
    # TODO ②
    X = df[FEATURE_NAMES].to_numpy()
    y = df[TARGET_NAME].to_numpy()

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = int(len(df) * test_ratio)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    y_train = y[train_idx]
    y_test = y[test_idx]
    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]

    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    X_train = (X_train_raw - mean) / std
    X_test = (X_test_raw - mean) / std
    return X_train, y_train, X_test, y_test, mean, std
    raise NotImplementedError("还没写 split_and_standardize")


# ============================================================
# 看一眼数据（已写好）
# ============================================================
def explore(df):
    print("=" * 66)
    print("数据长什么样")
    print("=" * 66)
    print(f"\n形状: {df.shape}   （{df.shape[0]} 个街区 × {df.shape[1]} 列）")
    print(f"\n前 5 行:")
    print(df.head().to_string())
    print(f"\n统计摘要:")
    print(df.describe().to_string())
    print("\n" + "=" * 66)
    print("注意看 std 那一行：不同特征的量纲差了多少倍？")
    print("这就是为什么必须做标准化。")
    print("=" * 66)


if __name__ == "__main__":
    df = load_data()
    explore(df)

    X_train, y_train, X_test, y_test, mean, std = split_and_standardize(df)
    print(f"\n划分完成:")
    print(f"  训练集 {X_train.shape}   测试集 {X_test.shape}")
    print(f"\n标准化之后（训练集）:")
    print(f"  每个特征的均值: {np.round(X_train.mean(axis=0), 10)}")
    print(f"  每个特征的标准差: {np.round(X_train.std(axis=0), 6)}")
    print(f"\n标准化之后（测试集）:")
    print(f"  每个特征的均值: {np.round(X_test.mean(axis=0), 4)}")
    print("  ↑ 不是精确的 0 —— 因为它用的是【训练集】的 mean/std。这是对的！")
    print("     如果这里是精确的 0，说明你不小心用了测试集自己的统计量（数据泄漏）。")
