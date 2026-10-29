# 交互 Demo

两个可以在终端里直接玩的 demo。

---

## ① 文本生成：Mini-GPT 莎士比亚

```bash
cd ~/projects/learn/DeepLearning/demo
conda activate learndl

python demo_text.py
```

交互命令：

| 输入 | 作用 |
|---|---|
| `ROMEO:` | 用这个开头生成 200 个字符 |
| `:temp 0.5` | 调温度（越低越保守） |
| `:len 400` | 调生成长度 |
| `:quit` | 退出 |

只生成一次：

```bash
python demo_text.py --once "ROMEO:"
```

模型：`04_transformer/minigpt_shakespeare.pt`
训练结果：train loss 1.04 / val loss 1.64 / perplexity 5.13

---

## ② 图像分类：Fashion-MNIST

```bash
python demo_image.py --index 42
```

或者用自己的图片：

```bash
python demo_image.py --image /path/to/photo.png

# 照片是白底、衣服是深色时，加 --invert 反色
python demo_image.py --image /path/to/photo.png --invert
```

> Fashion-MNIST 是**黑底、浅色衣服、28×28、居中**。
> 自己的照片如果背景/光照/构图差太多，识别会明显变差——这是训练分布不一致，不是模型坏了。

输出示例：

```
真实类别: Sneaker
预测结果:
  1. Sneaker        82.3%  ################################
  2. Ankle boot     12.1%  #####
  3. Bag             3.2%  #
```

模型：`02_fashion_mnist/cnn_improved.pt`
测试集准确率：92.25%

---

## 依赖

- `learndl` 环境
- 两个 checkpoint 必须存在：
  - `04_transformer/minigpt_shakespeare.pt`
  - `02_fashion_mnist/cnn_improved.pt`

---

## 常见问题

**Q: 报错 `FileNotFoundError`？**
A: 先跑对应的训练脚本生成 checkpoint。

**Q: 图片分类结果很差？**
A: Fashion-MNIST 是 28×28 灰度图，普通照片转过来会失真。
最好用数据集里的图（`--index`）演示。

**Q: 生成文本是乱码？**
A: 检查是否加载了正确的 checkpoint；temperature 不要超过 1.2。
