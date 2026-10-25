# 项目一 · 加州房价预测（线性回归）

用线性回归预测加州街区的房价中位数。

## 数据

California Housing（1990 年加州人口普查）· 20640 个街区 × 8 个特征

| 特征 | 含义 |
|---|---|
| MedInc | 街区收入中位数（万美元） |
| HouseAge | 房龄中位数 |
| AveRooms | 平均房间数 |
| AveBedrms | 平均卧室数 |
| Population | 街区人口 |
| AveOccup | 平均每户人数 |
| Latitude / Longitude | 经纬度 |

**目标**：`MedHouseVal` —— 房价中位数（单位 10 万美元）

## 目录

| 文件 | 内容 | 状态 |
|---|---|---|
| `data.py` | 加载 / 探索 / 标准化 / 划分 | 🔄 第 1 节 |
| `check1.py` | 第 1 节检查 | ✅ |
| `model.py` | 模型（前向 / 损失 / 梯度） | ⬜ 第 2 节 |
| `train.py` | 训练 + 曲线 | ⬜ 第 2 节 |
| `evaluate.py` | RMSE / R² / 残差分析 | ⬜ 第 2 节 |
| `predict.py` | 对新样本预测 | ⬜ 第 2 节 |
| `outputs/` | 训练曲线、预测对比图 | — |

## 运行

```bash
conda activate learndl
cd ~/projects/learn/DeepLearning/03_housing
python check1.py        # 第 1 节检查
```

## 过关标准

- 第 1 节：`check1.py` 2/2 通过
- 第 2 节：测试集 **R² ≥ 0.55**（线性回归在这个数据集上的合理水平）
