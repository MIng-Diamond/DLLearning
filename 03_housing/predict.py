"""
项目一 · 第 2 节 —— 预测
==========================

【本节最重要的工程概念】训练 / 推理 必须一致

模型是在【标准化后】的数据上训练的。推理时喂进去的数据，
必须经过【完全相同】的预处理 —— 用同一套 mean/std。

如果不做，模型看到的数字分布和训练时完全不一样，预测结果就是垃圾。
而且它不会报错，只会给你一个看似正常但其实错误的数字。

这是真实项目里最常见的线上事故之一：
    「训练时脚本里做了一步归一化，上线时忘了，模型准确率从 90% 掉到 10%」

所以正确的做法是把 mean/std 和模型参数【一起保存】，推理时一起加载。
"""

import numpy as np


def predict_raw(w, b, mean, std, raw_features):
    """给【原始量纲】的街区特征，预测房价（单位：10 万美元）。

    参数
        w, b         训练好的模型参数
        mean, std    【训练集】的均值和标准差（第 1 节返回的）
        raw_features (8,) 或 (M, 8)，原始单位
                     比如 [8.3, 41, 6.9, 1.0, 322, 2.5, 37.88, -122.23]
                     对应 [MedInc, HouseAge, AveRooms, AveBedrms,
                           Population, AveOccup, Latitude, Longitude]

    返回
        预测房价，单位 10 万美元。标量或 (M,)
        （不需要反标准化 —— 因为我们只标准化了 X，没有标准化 y）

    步骤（两行）
        1. 标准化：(raw_features - mean) / std
        2. 预测：  standardized @ w + b
    """
    # TODO ⑦
    features = (raw_features - mean) / std
    return features @ w + b
    raise NotImplementedError("还没写 predict_raw")
