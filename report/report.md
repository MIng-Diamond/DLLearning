# 深度学习网络手写练习阶段报告

**时间**：2026 年 10 月 11 日 — 10 月 28 日  
**代码**：<https://github.com/MIng-Diamond/DLLearning>

## 写在前面

关于本项目：模型结构和训练循环自己写；数据加载、优化器、autograd 用 PyTorch 现成 API。所以报告里不会出现“我调了某个库的模型”，但也不会假装所有东西都是从零实现的。哪些是自己写的、哪些是调用的，后面会分开说。

代码全部在仓库里，每个目录都有可运行的脚本和自检。报告里的图都是跑出来的。

## 一、完成情况

| 阶段 | 内容 | 结果 |
|---|---|---|
| 0 | NumPy 基础、softmax、线性回归、梯度检验 | 白纸测试 5/5 |
| 1 | NumPy 手推 MLP 反向传播；PyTorch 手搭 MLP | sin 拟合 loss 3.08 → 0.0031；Fashion-MNIST 86.88% |
| 项目零 | 加州房价线性回归 | R² = 0.6374，RMSE = 0.70 |
| 2 | 手写 conv2d / maxpool；PyTorch CNN | 90.24% → 92.25% |
| 3 | RNN / GRU / LSTM；手写 Attention；Transformer | 序列反转 100%；Mini-GPT val loss 1.64 |

截至 10 月 28 日，阶段 0 到阶段 3 已完成。阶段 4（VAE、DDPM、PPO、Behavior Cloning）还没开始。

## 二、练习方式

- 学习每种网络的基本理论
- 手写/PyTorch填模型和训练循环里的 TODO
- 用自检脚本检查形状、数值、收敛情况

学习过程中，较大部分时间花费在梳理tensor维度变换上

## 三、阶段 0–1：NumPy 与 MLP

阶段 0 内容包括 shape、索引、axis、广播、矩阵乘法，然后手写 softmax、线性回归、梯度下降和数值梯度检验。

阶段 1 用 NumPy 手推了一次两层 MLP 的反向传播，拟合 `sin(x)`。输入接 `Linear(1, 32) → ReLU → Linear(32, 1)`，反向传播手写。最后 loss 从 3.08 降到 0.0031，最优线性模型的 loss 是 0.167，差了两个数量级。

![图 1：NumPy 手写 MLP 拟合 sin(x)](../01_mlp/lesson05_result.png)

*图 1：NumPy 手写 MLP 拟合 sin(x)。左图是拟合结果，右图是 loss 曲线。*

此节我基本清楚反向传播等的底层逻辑，再切到 PyTorch 手搭就顺很多。

## 四、项目零：加州房价

数据是 California Housing，20,640 个街区，8 个特征，目标是房价中位数（单位 10 万美元）。标准化只用训练集的均值和方差，避免数据泄漏；划分是 80/20。

测试集结果：**R² = 0.6374，RMSE = 0.7048**（约 7 万美元）。

![图 2：加州房价线性回归诊断](../03_housing/outputs/diagnostics.png)

*图 2：加州房价线性回归诊断。右上角是测试集预测 vs 真实，R² = 0.6374；右下角是标准化后的特征权重。*

从图里能看到两个问题。第一，残差有明显的异方差：房价越高，误差越大。第二，真实房价在 5.0 附近有一条竖线，这是数据集本身把目标截断在 5.00001 造成的，不是模型的问题。

特征权重也能解释一些东西：收入中位数是正权重（+0.82），纬度和经度是负权重（-0.90 和 -0.87），和“沿海地区房价更高”的常识一致。

这个项目的重点不是把 R² 刷高，而是把整条链路跑通：数据划分、标准化只用训练集的原因、预测新样本时怎么复用同一套均值方差。后面几个项目都沿用了这套流程。

## 五、Fashion-MNIST

### 5.1 MLP

第一个版本是 MLP：

```
Flatten → Linear(784, 256) → ReLU → Linear(256, 10)
```

203,530 个参数，训练 20 轮，Adam，lr = 1e-3，weight decay = 1e-4。测试集准确率 86.88%。

这个结果不算差，但和最终 CNN（92.25%）比差了约 5 个百分点。因为 MLP 把 28×28 的图直接拉平，抛弃了二维结构，相邻像素的关系完全没有利用。

### 5.2 CNN

基础版 CNN 的结构是：

```
Conv(1→16, 3×3, padding=1) → BatchNorm(16) → ReLU → MaxPool(2)
Conv(16→32, 3×3, padding=1) → BatchNorm(32) → ReLU → MaxPool(2)
Flatten → Dropout(0.5) → Linear(1568, 10)
```

训练 10 轮，Adam，lr = 1e-3，batch 64，不做数据增强。测试集 90.24%（不同随机种子下大约在 90.2%~90.7% 之间）。比 MLP 高了约 3.4 个百分点。

之后继续优化：卷积加到三层（32 / 64 / 128 通道），每次卷积后加 BatchNorm，分类头前加 Dropout(0.5)；训练时加 RandomHorizontalFlip 和 RandomCrop(28, padding=4)；优化器用 Adam，学习率用 CosineAnnealingLR，训练 30 轮。测试集到了 92.25%。

![图 3：改进 CNN 的训练曲线](../02_fashion_mnist/cnn_improved_curves.png)

*图 3：改进 CNN 的 30 轮训练曲线。测试准确率最终 92.25%，训练和测试曲线贴得比较近。*

92.25% 在 Fashion-MNIST 上不算高。这一版的意义主要在于把常用技巧跑一遍，知道它们各自解决什么问题。

混淆矩阵更能说明问题：

![图 4：Fashion-MNIST 测试集混淆矩阵](../02_fashion_mnist/confusion_matrix_improved.png)

*图 4：测试集混淆矩阵。*

每类准确率：

| 类别 | 准确率 | 类别 | 准确率 |
|---|---|---|---|
| Trouser | 99.2% | Bag | 99.0% |
| Sneaker | 97.7% | Sandal | 97.5% |
| Ankle boot | 95.9% | Dress | 93.6% |
| T-shirt/top | 88.9% | Pullover | 88.8% |
| Coat | 87.8% | Shirt | 74.1% |

Shirt 是最差的，上半身衣物容易混淆，尤其是低分辨率灰度图。Trouser、Bag、Sandal 这些轮廓差别大的类别，准确率都在 97% 以上。

![图 5：分类正确的样本](../02_fashion_mnist/right_classified_improved.png)

*图 5：分类正确的样本。*

![图 6：分类错误的样本](../02_fashion_mnist/misclassified_improved.png)

*图 6：分类错误的样本。错例基本都集中在上半身衣物与鞋子之间。*

我还把第一层 32 个卷积核里的前 16 个画了出来：

![图 7：第一层卷积核](../02_fashion_mnist/cnn_filters_improved.png)

*图 7：第一层 32 个 3×3 卷积核中的前 16 个。*

能看出一些边缘和明暗响应，但没有很直观的产出。毕竟 3×3 的核容量很小，真正的语义特征在后几层。

## 六、序列与注意力

### 6.1 梯度消失

这一节做了一个对照实验：固定一个单层 RNN / GRU / LSTM，输入长度从 10 增加到 200，算最终 loss 对初始隐状态 `h0` 的梯度范数，取对数画图。

![图 8：梯度随序列长度的衰减](../03_rnn_attention/lesson09_result.png)

*图 8：不同循环结构下，梯度范数随序列长度的变化（纵轴对数刻度）。*

结果表明：

- RNN 在 T = 100 时梯度已经到 1e-40 量级，基本等于 0
- GRU 和默认初始化的 LSTM 慢一些，但 T = 100 时也到了 1e-18 ~ 1e-20
- 把 LSTM 的遗忘门 bias 初始化成 1 之后，T = 100 时梯度还在 1e-1 量级，T = 200 时约 1e-3，说明 LSTM 可缓解梯度消失。

原因分析：LSTM 的 cell state 有一条加法路径，遗忘门接近 1 时，梯度可以沿着这条路径传很远。默认初始化下遗忘门接近 0.5，则传播较短。

### 6.2 手写 Attention

尝试手写实现缩放点积注意力和 `F.scaled_dot_product_attention` 核对数值，最大误差小于 1e-6。

## 七、Transformer

### 7.1 400 字符上的 Mini-GPT

第一版 Mini-GPT 用的是 lesson11 里那段 400 字符的 Alice 文本。模型配置是 D = 128、4 头、3 层，参数量 605,352。训练 2000 步，loss 从 4.0 降到 0.05。

![图 9：400 字符语料上的 Mini-GPT 损失](../04_transformer/lesson11_result.png)

*图 9：400 字符语料上的 Mini-GPT 损失曲线。*

这一版把 Transformer 的零件跑通了：嵌入+位置编码、Pre-LN 残差块、FFN、因果 mask、训练循环、自回归生成。但 400 个字符实在太小，loss 降到 0.05 基本等于把训练文本背下来了，生成长一点的文本就会开始复制片段。

### 7.2 Encoder-Decoder 与交叉注意力

阶段 3 的过关任务是序列反转：

```
输入:  [3, 7, 1, 9, 2]
输出:  [2, 9, 1, 7, 3]
```

结构是标准的 Encoder-Decoder。Encoder 是双向自注意力，不 mask；Decoder 先做因果自注意力，再做交叉注意力，最后过 FFN。训练时用 Teacher Forcing，Decoder 输入是目标右移一位，前面补 BOS。

这个任务本身一行 `torch.flip` 就能做，虽然用神经网络小题大做。它的价值在于验证 seq2seq 的机制，验证 Encoder 对整段输入的编码效果，Decoder 是否可通过交叉注意力找到对应位置。

结果：16 条固定样本过拟合到 100%，256 条没见过的序列整句全对，也是 100%。

可视化交叉注意力头权重：

![图 10：序列反转任务学出的交叉注意力](../04_transformer/reversal_cross_attention.png)

*图 10：训练后的一个交叉注意力头。红框标出了理想的反向对齐位置，实际权重基本落在红框上。*

这个头的 argmax 是 `[5, 4, 3, 2, 1, 0]`，正好是反向对齐。算法中未提及反转逻辑，它是从数据中学出来的。

这一节发现诸多值得注意的地方：三种 mask 的用法、Q 和 K/V 长度不同、Teacher Forcing 的右移。详细的错误单独记在 `notes/debugging_notes.md` 里。

### 7.3 真实语料：tiny Shakespeare

有了 7.1 和 7.2 的基础，把语料换成 tiny Shakespeare（1,115,394 个字符，词表 65）。

模型配置：

| 项 | 值 |
|---|---|
| 参数量 | 4,772,417 |
| D | 256 |
| 层数 | 6 |
| 头数 | 8 |
| block_size | 128 |
| 训练步数 | 6000 |
| 优化器 | AdamW，lr 3e-4，warmup 200，cosine |
| 设备 | RTX 3060 Laptop，约 9 分钟 |

结果：

```
train loss  1.0399
val loss    1.6359
perplexity  5.13
```

![图 11：tiny Shakespeare 上的损失曲线](../04_transformer/lesson13_result.png)

*图 11：tiny Shakespeare 上的 train / val 损失。val loss 在 2000~2500 步最低（约 1.56），之后轻微回升。*

从曲线能看到一点过拟合：train loss 一直在降，val loss 在 2000 多步之后不再下降，最后回到 1.64。原因分析：110 万字符对 477 万参数来说偏少，模型训练到后期易过拟合。

生成效果（同一开头 `ROMEO:`，top-k = 40）：

```
temperature = 0.5
ROMEO:
I was born an appointed he hath been deliver'd.

Nurse:
What, what a sair, back and power to be list,
Which we shall become then there was like a life
...

temperature = 1.0
ROMEO:
I do that demand this friar slander; and, saw my weapon:
Look some betides, that preys greatness on him.

CAPULET:
That 'forgive, I hope, sovereign, be breast!
...
```

温度低的时候句子相对更通顺，但重复多；温度高的时候词汇更多样，错误词和语法错误也明显增加。这个规律和 softmax 的形状相似。温度低，分布尖，模型倾向于高频、安全的字符组合；温度高，分布平，采样空间变大，出错概率也变大。

模型的局限也很明显：字符级、477 万参数、没有指令微调。它只能续写莎士比亚风格的英文，不能对话。

## 八、不足

不足主要有四点：

1. 模型规模小。最大的模型是 477 万参数的字符级 Mini-GPT，离能用的语言模型还有距离。
2. CNN 没有做逐项消融。90.24% → 92.25% 是整体提升，但 BatchNorm、Dropout、数据增强各自贡献多少，没有单独测。
3. 没有测试集之外的泛化实验。Fashion-MNIST 的测试集和训练集同分布，换一个数据集（比如真实服装照片）效果会差很多。
4. 序列反转任务本身太简单，只能验证机制，不能说明模型有推理能力。

## 附录

### 仓库结构

```
DeepLearning/
├── 00_python_numpy/     NumPy、线性回归、梯度检验
├── 01_mlp/              NumPy 手写 MLP
├── 02_fashion_mnist/    MLP / CNN / 可视化
├── 03_housing/          加州房价完整项目
├── 03_rnn_attention/    RNN / LSTM / 梯度消失 / Attention
├── 04_transformer/      Transformer / Mini-GPT / Encoder-Decoder
├── demo/                交互 demo
├── notes/               踩坑记录
└── ROADMAP.md           学习路线和进度
```

