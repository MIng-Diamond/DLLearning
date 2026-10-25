"""
项目一 · 第 2 节 —— 模型
==========================

这四个函数你在 lesson03 已经写过一遍了。
这次是【默写 + 用在真实数据上】。写不出来就回去看 lesson03。

形状提醒（这次 X 有 8 个特征）：
    X : (N, 8)
    w : (8,)
    b : 标量
    y : (N,)
"""

import numpy as np


# ============================================================
def predict(X, w, b):
    """ŷ = X @ w + b

    X (N, 8)  →  返回 (N,)
    """
    # TODO ①
    return X @ w + b
    raise NotImplementedError("还没写 predict")


# ============================================================
def mse_loss(y_pred, y_true):
    """均方误差 L = mean((y_pred - y_true) ** 2)，返回标量"""
    # TODO ②
    return np.mean((y_pred- y_true) ** 2)
    raise NotImplementedError("还没写 mse_loss")


# ============================================================
def compute_gradients(X, y_true, y_pred):
    """返回 (dw, db)

        d  = y_pred - y_true       # (N,)   误差（注意是预测减真实）
        dw = (2/N) * X.T @ d       # (8,)   和 w 同形状
        db = (2/N) * np.sum(d)     # 标量

    这是你在阶段 0 通关测试里默写过的东西，这次用在 8 维数据上。
    """
    # TODO ③
    d = y_pred - y_true
    dw = 2 / y_true.shape[0] * X.T @ d
    db = 2 / y_true.shape[0] * np.sum(d)
    return dw, db
    raise NotImplementedError("还没写 compute_gradients")


# ============================================================
def train(X, y, lr=0.1, epochs=2000, w=None, b=0.0):
    """梯度下降训练。返回 (w, b, losses)

    四步循环：
        ① y_pred = predict(X, w, b)
        ② loss   = mse_loss(y_pred, y)
        ③ dw, db = compute_gradients(X, y, y_pred)
        ④ w -= lr * dw ;  b -= lr * db

    losses 要记录每一轮的损失（画曲线用）。
    w=None 时要自己初始化：np.zeros(特征个数)
    """
    # TODO ④
    if w is None:
        w = np.zeros((X.shape[1],))
    losses = []
    for epoch in range(epochs):
        y_pred = predict(X, w, b)
        loss = mse_loss(y_pred, y)
        losses.append(loss)
        dw, db = compute_gradients(X, y, y_pred)
        w = w - lr * dw
        b = b - lr * db
    return (w, b, losses)
    raise NotImplementedError("还没写 train")
