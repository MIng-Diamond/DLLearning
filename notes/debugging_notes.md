# 踩坑记录

这份记录不属于报告正文，单独放在这里，方便以后查。

下面这些错误，基本都是形状或 mask 的问题，不是数学不会。

## 错误清单

| 问题 | 现象 | 原因 | 修复 |
|---|---|---|---|
| 位置编码写错 | `AttributeError: 'PositionalEncoding' object has no attribute 'pre'` | 把 `pe` 打成了 `pre` | 改成 `self.pe` |
| 位置编码用乘 | 模型能跑，但 loss 降不下去 | 位置编码应该加到嵌入上，不是乘 | `x + pe` |
| 交叉注意力拆头没赋值 | 形状检查全过，泛化只有 12.6% | `q.reshape(...).transpose(...)` 返回新张量，没赋值回去 | `q = q.reshape(...)` |
| `masked_fill` 参数写成 `0,0` | `TypeError: got (Tensor, int, int)` | Python 小数点是 `.`，逗号是参数分隔符 | 改成 `0.0` |
| `torch.cat` 少了 `dim=1` | `Sizes of tensors must match except in dimension 0` | 默认沿 dim=0 拼接 | 加 `dim=1` |
| `ys` 没放到 GPU | `index is on cpu, different from other tensors on cuda` | `torch.full` 默认在 CPU | 加 `device=x.device` |
| `B, _ = x.shape` 写在取数据之前 | `UnboundLocalError: x` | 顺序写反了 | 先 `x, y = get_batch()`，再取 `B` |
| bool mask 语义搞反 | 被 mask 的位置反而被关注 | SDPA 里 `True` = 允许，`nn.MultiheadAttention` 里 `True` = 屏蔽 | 查文档，别凭记忆 |

## 最隐蔽的一个

交叉注意力的 `reshape` 和 `transpose` 没有赋值回去，`q/k/v` 还是三维，`F.scaled_dot_product_attention` 把最后一维当成了序列维。

形状检查全过，过拟合 16 条样本也能到 100%，只有“泛化到新序列”这一项暴露了问题。这也说明为什么自检里要留一个泛化测试：只过拟合几条固定样本，是发现不了这种错误的。

## 调试方法

后来固定成三步：

1. 每个组件用极小的尺寸单独测（B=2, T=3, D=8），先过形状
2. 自底向上，一个组件过了再测下一个
3. 第一个失败就是当前要修的地方，不要一次改多个地方

这套方法在 lesson12 里救命了。如果直接跑完整模型，报错会出现在很深的地方，根本不知道是哪一层的形状错了。
