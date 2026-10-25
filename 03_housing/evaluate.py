"""
项目一 · 第 2 节 —— 评估
==========================

【本节的新东西】怎么衡量一个回归模型「有多好」？

MSE 有个问题：它的单位是「目标变量的平方」。
你的房价单位是 10 万美元，MSE 的单位就成了「10万美元²」——没法跟人解释。

所以要换成两个能说人话的指标。
"""

import numpy as np


# ============================================================
def rmse(y_pred, y_true):
    """均方根误差 Root Mean Squared Error

        RMSE = sqrt( mean( (y_pred - y_true)² ) )

    就是 MSE 开个根号。好处是【单位回到了目标变量本身】：
    如果 y 的单位是 10 万美元，RMSE = 0.7 就意味着
    「平均来说，预测值偏离真实值 0.7 个 10 万美元，也就是约 7 万美元」。

    提示：一行，np.sqrt + np.mean。
    """
    # TODO ⑤
    return np.sqrt(np.mean((y_pred - y_true) ** 2))
    raise NotImplementedError("还没写 rmse")


# ============================================================
def r2_score(y_pred, y_true):
    """决定系数 R²（读作 R squared）

        SS_res = Σ (y_true - y_pred)²     ← 残差平方和：我的模型错了多少
        SS_tot = Σ (y_true - ȳ)²          ← 总平方和：  数据本身波动多大
        R² = 1 - SS_res / SS_tot

    【怎么理解这个公式】

    想象一个「傻瓜模型」：不管输入是什么，永远预测训练集的平均值 ȳ。
    它的误差就是 SS_tot。

    再看看你的模型：误差是 SS_res。

    所以 SS_res / SS_tot 就是「你的误差 ÷ 傻瓜的误差」：
        = 0.0  →  你和傻瓜一样准   →  R² = 1     完美
        = 0.4  →  你的误差只有傻瓜的 40%  →  R² = 0.6   （解释力 60%）
        = 1.0  →  你和傻瓜一样差   →  R² = 0
        > 1.0  →  你比傻瓜还差     →  R² < 0     （真的会发生！）

    所以 R² 的含义是：**你的模型解释了目标变量多少比例的波动。**

    ⚠️ 注意：分母里的 ȳ 是【y_true 自己的均值】，不用另外传进来。

    提示：两行，两个 np.sum。
    """
    # TODO ⑥
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

    raise NotImplementedError("还没写 r2_score")


# ============================================================
# 残差分析（已写好，直接调用）
# ============================================================
def residual_report(y_pred, y_true, n_worst=5):
    """残差 = 真实值 - 预测值。看看模型在哪些样本上错得最离谱。"""
    resid = y_true - y_pred
    print(f"    残差均值 : {resid.mean():+.4f}   （越接近 0 越好，说明没有系统性偏差）")
    print(f"    残差标准差: {resid.std():.4f}")
    print(f"    最大高估 : {resid.min():+.4f}   （模型预测得比真实高了这么多）")
    print(f"    最大低估 : {resid.max():+.4f}")

    worst = np.argsort(-np.abs(resid))[:n_worst]
    print(f"\n    错得最离谱的 {n_worst} 个样本:")
    for i in worst:
        print(f"      样本 {i:5d}  真实 {y_true[i]:.3f}  预测 {y_pred[i]:.3f}  "
              f"残差 {resid[i]:+.3f}")
