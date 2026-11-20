# 神经网络与大语言模型学习指南

> 从数学基础出发，通过简单的 Python 和 PyTorch 实验，逐步理解机器学习、神经网络以及大语言模型（LLM）的学习原理。

> 本文是详细版学习笔记，完整原理与逐模块说明都在这里。
> 只想快速上手请先看 [../README.md](../README.md)，每个源文件开头也带有一段同主题的模块说明。

## 一、项目介绍

本项目主要用于学习人工智能、机器学习和神经网络的基础知识。

不以直接调用大模型 API 为目标，而是从底层原理出发，理解：

- 机器学习是什么？
- 神经网络是如何工作的？
- 模型为什么能够通过数据进行学习？
- 梯度、导数、反向传播和优化器之间有什么关系？
- 大语言模型是如何学习语言、理解上下文并生成文本的？
- Transformer、Attention、Embedding、Token 等概念分别是什么？

通过数学推导、代码分析和实际 Demo，逐步建立对神经网络和大语言模型的整体认识。

### 学习目标

完成本项目后，希望能够：

1. 理解机器学习的基本概念。
2. 掌握神经网络训练的基本流程。
3. 理解导数、梯度、链式法则和梯度下降。
4. 理解损失函数、反向传播和优化器。
5. 使用 NumPy 手动实现简单的机器学习算法。
6. 使用 PyTorch 构建、训练和评估简单神经网络。
7. 理解大语言模型的基本训练原理。
8. 对 Transformer、Attention 和 Token 有初步认识。

---

## 二、整体知识结构

```text
人工智能（AI）
    |
    └── 机器学习（Machine Learning）
            |
            └── 深度学习（Deep Learning）
                    |
                    └── 神经网络（Neural Network）
                            |
                            └── Transformer
                                    |
                                    └── 大语言模型（LLM）
```

需要注意：这是一个便于入门理解的知识关系图，并不意味着所有人工智能技术都必须经过这条路径。

### 学习路线

```text
第一阶段：数学基础
    |
    ├── 函数
    ├── 导数
    ├── 偏导数
    ├── 梯度
    ├── 矩阵与向量
    ├── 概率与统计
    └── 对数与指数
    |
第二阶段：机器学习基础
    |
    ├── 训练数据与测试数据
    ├── 模型与参数
    ├── 损失函数
    ├── 梯度下降
    └── 模型评估
    |
第三阶段：神经网络基础
    |
    ├── 神经元
    ├── 线性层
    ├── 激活函数
    ├── 前向传播
    ├── 反向传播
    └── 优化器
    |
第四阶段：大语言模型基础
    |
    ├── Token
    ├── Embedding
    ├── Transformer
    ├── Attention
    ├── 预训练
    ├── 微调
    └── 对齐训练
```

---

## 三、数学基础

数学是理解神经网络的基础，但不需要一开始就掌握所有高等数学知识。

建议重点理解以下内容。

### 3.1 函数

函数可以理解为一种映射关系：输入一个值，按照一定规则计算，得到一个输出值。

例如：

\[
y = 2x + 1
\]

当：

\[
x = 3
\]

那么：

\[
y = 2 \times 3 + 1 = 7
\]

在机器学习中，模型本质上也是一个函数。

\[
y_{pred} = f(x)
\]

其中：

- \(x\)：输入数据。
- \(f\)：模型。
- \(y_{pred}\)：模型预测结果。

模型训练的过程，就是不断调整函数中的参数，让预测结果尽可能接近真实结果。

### 3.2 导数

导数描述函数在某个位置的变化率。

例如：

\[
y = x^2
\]

它的导数为：

\[
\frac{dy}{dx} = 2x
\]

当 \(x=3\) 时：

\[
\frac{dy}{dx}=6
\]

可以理解为：在 \(x=3\) 附近，输入每发生一个很小的变化，输出大约会发生 6 倍的变化。

在机器学习中，导数可以帮助我们理解：

**模型参数发生变化时，损失函数会如何变化？**

### 3.3 偏导数

当一个函数存在多个变量时，可以分别研究每个变量对函数的影响，这就是偏导数。

例如：

\[
L = w_1x_1 + w_2x_2
\]

分别对两个权重求偏导：

\[
\frac{\partial L}{\partial w_1}=x_1
\]

\[
\frac{\partial L}{\partial w_2}=x_2
\]

在神经网络中，一个模型通常存在大量权重参数，因此需要计算每个参数对应的偏导数。

### 3.4 梯度

梯度是由多个偏导数组成的向量。

例如：

\[
L = w_1^2 + 2w_2^2
\]

它的梯度为：

\[
\nabla L =
\begin{bmatrix}
2w_1 \\
4w_2
\end{bmatrix}
\]

梯度告诉我们：

- 每个参数对 Loss 的影响有多大。
- Loss 增长最快的方向是什么。
- 如果希望降低 Loss，参数应该朝哪个方向调整。

### 3.5 链式法则

链式法则是神经网络反向传播的重要数学基础。

假设：

\[
y = u^2
\]

\[
u = 2x+1
\]

那么：

\[
\frac{dy}{dx}
=
\frac{dy}{du}
\times
\frac{du}{dx}
\]

代入：

\[
\frac{dy}{dx}
=
2u \times 2
=
4(2x+1)
\]

神经网络由多个计算层组成，最终 Loss 与前面每一层的参数都有关系。

反向传播就是利用链式法则，从 Loss 开始，逐层计算每个参数的梯度。

### 3.6 向量与矩阵

向量可以理解为一组有序数字。

例如：

\[
x =
\begin{bmatrix}
0.1 \\
0.8 \\
0.3
\end{bmatrix}
\]

矩阵可以理解为一个二维数字表格。

例如：

\[
W =
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
\]

神经网络中的输入数据、权重参数和输出结果，通常都会以向量或矩阵的形式进行计算。

一个简单的线性层：

\[
y = Wx+b
\]

其中：

- \(x\)：输入向量。
- \(W\)：权重矩阵。
- \(b\)：偏置。
- \(y\)：输出向量。

### 3.7 概率与统计

概率用于描述事件发生的可能性。

例如：

\[
P(A)=0.8
\]

表示事件 A 发生的概率为 80%。

在分类任务中，模型通常会输出多个类别的分数，再通过 Softmax 转换为概率：

\[
P_i =
\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

其中：

- \(z_i\)：第 i 个类别的原始预测分数。
- \(P_i\)：第 i 个类别对应的概率。

所有类别概率之和为 1。

### 3.8 对数与交叉熵

交叉熵是分类任务中常用的损失函数。

对于一个样本：

\[
Loss=-\log(P_{true})
\]

其中 \(P_{true}\) 是模型给真实类别分配的概率。

例如：

| 真实类别的预测概率 | Loss |
|---|---:|
| 0.9 | 0.105 |
| 0.5 | 0.693 |
| 0.2 | 1.609 |
| 0.1 | 2.303 |

可以发现：

模型给真实类别的概率越高，Loss 越低。

这就是为什么训练分类模型时，我们希望交叉熵损失不断降低。

---

## 四、机器学习基础

### 4.1 什么是机器学习？

传统编程通常是：

```text
输入数据 + 人工编写的规则
          |
          v
        计算机
          |
          v
        输出结果
```

机器学习则是：

```text
输入数据 + 正确答案
          |
          v
       机器学习
          |
          v
       学习规律
          |
          v
        模型
          |
          v
    输入新的数据
          |
          v
      预测结果
```

传统编程需要开发人员明确编写规则。

机器学习则是通过大量数据和正确答案，让模型自动调整参数，学习输入与输出之间的关系。

### 4.2 模型是什么？

模型可以理解为一个具有可调整参数的数学函数。

例如：

\[
y = w_1x_1+w_2x_2+b
\]

其中：

- \(x_1,x_2\)：输入特征。
- \(w_1,w_2\)：权重。
- \(b\)：偏置。

模型训练的核心就是找到合适的权重和偏置。

### 4.3 什么是 Loss？

Loss（损失）用于衡量模型预测结果与真实结果之间的差距。

例如：

真实房价：

\[
y=100
\]

模型预测：

\[
y_{pred}=80
\]

使用均方误差：

\[
Loss=(100-80)^2=400
\]

如果预测结果变为 95：

\[
Loss=(100-95)^2=25
\]

Loss 越小，说明预测结果越接近真实结果。

不同任务会使用不同的损失函数，例如：

- 回归任务：MSE（均方误差）。
- 分类任务：Cross Entropy（交叉熵）。

### 4.4 什么是梯度下降？

梯度下降是一种常见的参数优化方法。

它的核心思想是：

根据 Loss 对参数的梯度，沿着能够降低 Loss 的方向调整参数。

更新公式：

\[
w_{new}=w_{old}-\alpha\frac{\partial Loss}{\partial w}
\]

其中：

- \(w\)：模型参数。
- \(\alpha\)：学习率。
- \(\frac{\partial Loss}{\partial w}\)：参数对应的梯度。

学习率决定每次调整参数的步长。

学习率太大，可能导致 Loss 波动甚至无法收敛。

学习率太小，则可能导致训练速度过慢。

---

## 五、神经网络基础

### 5.1 什么是神经网络？

神经网络是一种由多个计算单元组成的模型。

一个简单的神经网络：

```text
输入层          隐藏层          输出层

 x1  ───────┐
            ├──── h1 ─────┐
 x2  ───────┤             ├──── y1
            ├──── h2 ─────┤
 x3  ───────┤             ├──── y2
            └──── h3 ─────┘
```

每个神经元会接收输入，进行加权计算，再通过激活函数处理，最终将结果传递给下一层。

### 5.2 神经元的计算过程

一个简单神经元：

\[
z=w_1x_1+w_2x_2+b
\]

然后经过激活函数：

\[
a=f(z)
\]

其中：

- \(x\)：输入。
- \(w\)：权重。
- \(b\)：偏置。
- \(f\)：激活函数。
- \(a\)：神经元输出。

常见激活函数：

| 激活函数 | 作用 |
|---|---|
| Sigmoid | 将输出映射到 0～1 |
| Tanh | 将输出映射到 -1～1 |
| ReLU | 小于 0 的值变为 0，大于 0 的值保持不变 |
| GELU | Transformer 中常见的激活函数 |

### 5.3 模型训练的完整流程

模型训练可以概括为五个核心步骤：

**第一步：前向传播**

将输入数据传入模型，计算预测结果。

```python
y_pred = model(x)
```

**第二步：计算损失**

通过损失函数比较预测结果和真实标签。

```python
loss = nn.functional.cross_entropy(y_pred, y)
```

**第三步：反向传播**

根据 Loss，通过链式法则计算模型参数的梯度。

```python
loss.backward()
```

**第四步：更新参数**

优化器根据梯度调整模型的权重和偏置。

```python
optim.step()
```

**第五步：重复训练**

不断重复前面的过程，让模型逐渐学习数据中的规律。

```python
for epoch in range(epoch_num):
    for x, y in train_loader:
        optim.zero_grad()

        y_pred = model(x)
        loss = nn.functional.cross_entropy(y_pred, y)

        loss.backward()
        optim.step()
```

注意：

- 反向传播负责计算梯度。
- 优化器负责更新参数。
- 梯度清零用于防止不同批次的梯度意外累积。

---

## 六、大语言模型是如何学习的？

大语言模型（Large Language Model，LLM）是使用大量文本数据训练出来的深度学习模型。

例如 GPT、DeepSeek、Qwen、Llama 等。

它们通常使用 Transformer 架构，通过学习大量文本中的统计规律，获得语言建模能力。

### 6.1 大模型学习的核心思想

可以先用一句话理解：

**给模型大量文本，让它不断学习根据前面的内容预测下一个 Token。**

例如：

输入：

```text
今天天气非常
```

模型需要预测下一个 Token。

可能的结果：

```text
好
```

也可能是：

```text
不错
```

模型会根据上下文，为不同候选 Token 计算概率。

训练时，如果真实的下一个 Token 是“好”，模型就会根据真实答案计算 Loss，再通过反向传播调整参数。

重复大量训练后，模型逐渐学习到语言、语法、语义、上下文以及知识之间的统计关联。

### 6.2 Token 是什么？

Token 是大语言模型处理文本的基本单位。

Token 不一定等于一个汉字、一个单词或者一个完整词语。

例如：

```text
我喜欢学习人工智能
```

可能被拆分为：

```text
我 | 喜欢 | 学习 | 人工 | 智能
```

实际如何拆分，取决于模型使用的 Tokenizer。

模型并不是直接理解原始字符串，而是先将文本转换为 Token ID。

```text
原始文本
   |
   v
Tokenizer
   |
   v
Token ID
   |
   v
Embedding
   |
   v
Transformer
```

### 6.3 Embedding 是什么？

Embedding 是将离散的 Token ID 转换为连续的向量表示。

例如：

```text
Token：苹果
Token ID：1024
Embedding：[0.12, 0.36, -0.25, ...]
```

每个 Token 会对应一个向量。

向量可以作为神经网络进行数学计算的输入。

通过训练，Embedding 会逐渐形成能够反映语言使用规律的表示。

### 6.4 Transformer 是什么？

Transformer 是现代大语言模型广泛使用的神经网络架构。

它的核心组成包括：

- Embedding：将 Token 转换成向量。
- Positional Information：提供序列位置信息。
- Self-Attention：计算不同 Token 之间的关联。
- Feed Forward Network：对特征进行进一步变换。
- Residual Connection：残差连接。
- Layer Normalization：层归一化。

简化结构：

```text
输入文本
   |
   v
Tokenizer
   |
   v
Token Embedding
   |
   v
位置编码 / 位置信息
   |
   v
Transformer Block
   |
   ├── Multi-Head Attention
   |
   ├── Feed Forward Network
   |
   ├── Residual Connection
   |
   └── Layer Normalization
   |
   v
输出层
   |
   v
预测下一个 Token 的概率
```

实际 Transformer 的具体结构和组件排列会因模型类型而有所不同。

### 6.5 Attention 是什么？

Attention（注意力机制）是 Transformer 的核心技术之一。

它让模型在处理当前 Token 时，能够根据上下文关注其他相关 Token。

例如：

```text
小明把书放在桌子上，因为它很重。
```

模型需要根据上下文判断“它”可能指什么。

Attention 会计算不同 Token 之间的关联权重，使模型能够综合上下文信息。

Self-Attention 的核心公式：

\[
Attention(Q,K,V)
=
Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

其中：

- Q：Query，查询向量。
- K：Key，键向量。
- V：Value，值向量。
- \(d_k\)：Key 向量的维度。

可以先简单理解为：

1. Q 和 K 计算关联程度。
2. Softmax 将关联分数转换成权重。
3. 使用权重对 V 进行加权求和。
4. 得到融合上下文信息的新向量。

### 6.6 大模型是如何训练的？

大模型训练通常可以分为几个阶段。

**阶段一：预训练（Pre-training）**

使用大量文本数据训练模型。

主要目标通常是预测下一个 Token。

例如：

```text
输入：人工智能正在改变
目标：世界
```

模型根据输入预测下一个 Token，然后计算预测结果与真实 Token 之间的交叉熵损失。

训练过程：

```text
大量文本
   |
   v
Tokenizer
   |
   v
Transformer
   |
   v
预测下一个 Token
   |
   v
计算 Cross Entropy Loss
   |
   v
反向传播
   |
   v
更新模型参数
   |
   v
重复训练
```

经过大量训练，模型会学习到丰富的语言模式和知识关联。

**阶段二：监督微调（SFT）**

预训练主要学习文本分布，而监督微调通常会使用指令和回答组成的数据，让模型学习如何按照指令回答问题。

例如：

```json
{
  "instruction": "解释什么是神经网络",
  "output": "神经网络是一种由多个计算单元组成的机器学习模型..."
}
```

通过大量类似的数据，让模型逐渐具备指令跟随和问答能力。

**阶段三：偏好对齐（Alignment）**

希望模型生成的回答不仅合理，还符合人类偏好和安全要求。

常见方法包括：

- RLHF：基于人类反馈的强化学习。
- DPO：直接偏好优化。

不同模型采用的训练流程可能不同，并不是所有模型都会严格使用完全相同的阶段。

### 6.7 大模型训练和普通神经网络训练有什么关系？

两者的底层数学原理是相通的。

| 对比项目 | 普通神经网络 | 大语言模型 |
|---|---|---|
| 输入 | 数值、图片等 | Token ID / 向量 |
| 模型 | 线性层、CNN 等 | Transformer 等 |
| 输出 | 分类结果、预测数值 | Token 概率分布 |
| 损失函数 | MSE、交叉熵等 | 通常使用交叉熵 |
| 反向传播 | 计算参数梯度 | 计算参数梯度 |
| 优化器 | SGD、Adam 等 | AdamW 等常见优化器 |
| 训练目标 | 学习特定任务 | 学习语言规律及任务能力 |

**大模型并没有脱离神经网络训练的基本原理。它仍然通过前向传播、计算 Loss、反向传播和优化器更新参数来学习，只是模型规模、训练数据和计算复杂度大幅提高。**

---

## 七、两个入门 Demo

本项目准备两个 Demo，分别从数学原理和 PyTorch 框架两个角度理解模型训练。

代码已落到仓库中，可直接运行：

```bash
python 04-demos/demo01-numpy-linear-regression.py
python 04-demos/demo02-pytorch-classification.py
```

### Demo 01：使用 NumPy 手动实现线性回归

**学习目标：**

不使用 PyTorch，自己实现一个最简单的机器学习模型，理解：

- 模型参数是什么。
- Loss 如何计算。
- 导数如何计算。
- 梯度下降如何更新参数。
- 模型如何逐渐拟合数据。

#### 任务描述

假设存在如下规律：

\[
y=2x+3
\]

构造一些训练数据，让模型自己学习出：

\[
y_{pred}=wx+b
\]

最终希望训练得到：

\[
w\approx 2,\quad b\approx 3
\]

#### 完整代码

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成训练数据
np.random.seed(42)

x = np.linspace(0, 10, 100)
y = 2 * x + 3 + np.random.normal(0, 1, 100)

# 初始化模型参数
w = 0.0
b = 0.0

# 学习率
learning_rate = 0.01

# 训练轮数
epoch_num = 1000

loss_list = []

for epoch in range(epoch_num):

    # 1. 前向传播
    y_pred = w * x + b

    # 2. 计算 Loss（均方误差）
    loss = np.mean((y_pred - y) ** 2)

    # 3. 计算梯度
    dw = np.mean(2 * (y_pred - y) * x)
    db = np.mean(2 * (y_pred - y))

    # 4. 更新参数
    w = w - learning_rate * dw
    b = b - learning_rate * db

    loss_list.append(loss)

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch: {epoch + 1}, "
            f"Loss: {loss:.4f}, "
            f"w: {w:.4f}, "
            f"b: {b:.4f}"
        )

print("最终权重 w：", w)
print("最终偏置 b：", b)

# 绘制 Loss 曲线
plt.plot(loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

# 绘制拟合结果
plt.scatter(x, y, label="Original Data")
plt.plot(x, w * x + b, label="Prediction")
plt.legend()
plt.show()
```

#### 重点理解

模型：

\[
y_{pred}=wx+b
\]

损失函数：

\[
Loss=\frac{1}{n}\sum_{i=1}^{n}(y_{pred}-y)^2
\]

权重梯度：

\[
\frac{\partial Loss}{\partial w}
=
\frac{2}{n}\sum_{i=1}^{n}(y_{pred}-y)x
\]

偏置梯度：

\[
\frac{\partial Loss}{\partial b}
=
\frac{2}{n}\sum_{i=1}^{n}(y_{pred}-y)
\]

参数更新：

\[
w=w-\alpha\frac{\partial Loss}{\partial w}
\]

\[
b=b-\alpha\frac{\partial Loss}{\partial b}
\]

这个 Demo 的价值是：不依赖深度学习框架，直接通过数学公式理解模型训练的全过程。

---

### Demo 02：使用 PyTorch 实现五分类神经网络

**学习目标：**

通过一个简单的五分类任务，理解：

- 神经网络模型的定义。
- Linear 线性层。
- 输入和输出维度。
- CrossEntropyLoss。
- 反向传播。
- Adam 优化器。
- 训练集、测试集和准确率。

#### 任务描述

输入一个五维向量：

```python
x = [0.1, 0.8, 0.3, 0.4, 0.2]
```

要求模型预测最大值所在的位置。

最大值为：

```text
0.8
```

对应下标：

```text
1
```

最终展示位置：

```text
2
```

这里需要注意：模型内部使用 0～4 作为类别标签，面向用户展示时才加 1。

#### 完整代码

```python
import torch
import torch.nn as nn
import numpy as np


# 定义神经网络
class TorchModel(nn.Module):

    def __init__(self):
        super(TorchModel, self).__init__()

        # 5个输入特征，5个输出类别
        self.linear = nn.Linear(5, 5)

    def forward(self, x):
        return self.linear(x)


# 随机生成五维向量
def random_simulation():

    x = np.random.random(5)

    # 获取最大值所在下标，范围0～4
    y = x.argmax()

    return x, y


# 构建数据集
def build_dataset(total_sample_num):

    total_x = []
    total_y = []

    for i in range(total_sample_num):

        x, y = random_simulation()

        total_x.append(x)
        total_y.append(y)

    return (
        torch.tensor(total_x, dtype=torch.float32),
        torch.tensor(total_y, dtype=torch.long)
    )


# 评估模型
def evaluate(model, test_x, test_y):

    model.eval()

    with torch.no_grad():

        y_pred = model(test_x)

        predicted = torch.argmax(y_pred, dim=1)

        accuracy = (predicted == test_y).float().mean().item()

    return accuracy


def main():

    # 训练参数
    epoch_num = 50
    batch_size = 32
    train_sample = 5000
    test_sample = 1000
    learning_rate = 0.001

    # 创建模型
    model = TorchModel()

    # 创建优化器
    optim = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # 构建训练集和测试集
    train_x, train_y = build_dataset(train_sample)
    test_x, test_y = build_dataset(test_sample)

    # 开始训练
    for epoch in range(epoch_num):

        model.train()

        total_loss = 0

        # 每个 epoch 遍历全部训练数据
        for start in range(0, train_sample, batch_size):

            end = min(start + batch_size, train_sample)

            x = train_x[start:end]
            y = train_y[start:end]

            # 梯度清零
            optim.zero_grad()

            # 前向传播
            y_pred = model(x)

            # 计算损失
            loss = nn.functional.cross_entropy(y_pred, y)

            # 反向传播
            loss.backward()

            # 更新参数
            optim.step()

            total_loss += loss.item() * len(y)

        avg_loss = total_loss / train_sample

        # 评估模型
        accuracy = evaluate(model, test_x, test_y)

        print(
            f"Epoch: {epoch + 1}, "
            f"Loss: {avg_loss:.4f}, "
            f"Accuracy: {accuracy:.4f}"
        )

    # 单个样本预测
    test_input = torch.tensor(
        [[0.1, 0.8, 0.3, 0.4, 0.2]],
        dtype=torch.float32
    )

    model.eval()

    with torch.no_grad():

        output = model(test_input)

        predicted = torch.argmax(output, dim=1).item()

    print("预测最大值位置：", predicted + 1)


if __name__ == "__main__":
    main()
```

#### 重点理解

模型结构：

```python
self.linear = nn.Linear(5, 5)
```

表示：

- 输入维度为 5。
- 输出维度为 5。
- 权重参数为 25 个。
- 偏置参数为 5 个。
- 总参数为 30 个。

训练流程：

```python
optim.zero_grad()

y_pred = model(x)

loss = nn.functional.cross_entropy(y_pred, y)

loss.backward()

optim.step()
```

对应：

| 代码 | 作用 |
|---|---|
| `model(x)` | 前向传播 |
| `cross_entropy()` | 计算损失 |
| `loss.backward()` | 反向传播，计算梯度 |
| `optim.step()` | 更新参数 |
| `optim.zero_grad()` | 梯度清零 |

这个 Demo 的价值是：理解如何使用 PyTorch 搭建神经网络，并通过训练数据让模型自动调整参数。

注意：这个任务实际上可以直接使用 `argmax()` 得到答案。使用神经网络是为了学习训练机制，而不是为了提高这个特定任务的效率。

---

## 八、推荐学习顺序

建议按照以下顺序学习，不要一开始就直接研究大模型源码。

| 顺序 | 学习内容 | 重点 |
|---|---|---|
| 1 | 函数、导数 | 理解函数变化和变化率 |
| 2 | 偏导数、梯度 | 理解多个参数如何影响 Loss |
| 3 | 链式法则 | 理解反向传播的数学基础 |
| 4 | 向量、矩阵 | 理解神经网络的数据计算 |
| 5 | NumPy 线性回归 | 手动实现梯度下降 |
| 6 | PyTorch 基础 | 理解模型、Loss 和优化器 |
| 7 | 神经网络 | 理解多层网络和激活函数 |
| 8 | Softmax、交叉熵 | 理解分类模型训练 |
| 9 | Token、Embedding | 理解语言模型输入 |
| 10 | Transformer、Attention | 理解大模型架构 |
| 11 | 大模型训练 | 理解预训练、微调和对齐 |

---

## 九、推荐学习资源

以下资源可以作为进一步学习的入口：

### 数学基础

- [3Blue1Brown - Essence of Calculus](https://www.3blue1brown.com/topics/calculus)：通过动画理解微积分。
- [3Blue1Brown - Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)：理解线性代数、向量和矩阵。

### Python 与深度学习

- [NumPy 官方文档](https://numpy.org/doc/)
- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)

### 神经网络与大模型

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)：通过图示理解 Transformer。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：Transformer 原始论文。
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)：通过代码理解 Transformer。
- [Hugging Face Course](https://huggingface.co/learn)：学习 Tokenizer、Transformer 和模型使用。

---

## 十、项目目录结构

以下就是本仓库的实际结构，每个文件都可直接运行或阅读。

```text
neural-network/
│
├── README.md                        简明入口：环境、运行方式、模块索引
├── requirements.txt                 依赖清单
│
├── docs/
│   └── detailed-guide.md            本文件：完整原理与逐模块说明
│
├── 01-mathematics/                  第一阶段：数学基础
│   ├── 01-function.py               函数与"模型即参数化映射"
│   ├── 02-derivative.py             导数、差商逼近、变化率
│   ├── 03-gradient.py               偏导数、梯度方向、梯度下降
│   ├── 04-chain-rule.py             链式法则与反向传播的数学铺垫
│   └── 05-matrix.py                 向量矩阵、线性层形状、内积相似度
│
├── 02-machine-learning/             第二阶段：机器学习基础
│   ├── 01-linear-regression.py      最小完整训练闭环与模型评估
│   └── 02-gradient-descent.py       学习率影响、批量/随机/小批量对比
│
├── 03-neural-network/               第三阶段：神经网络基础
│   ├── 01-linear-layer.py           线性层与四种激活函数
│   ├── 02-forward-propagation.py    逐层前向传播与中间结果缓存
│   ├── 03-backpropagation.py        手算梯度并用数值差商校验
│   └── 04-classification.py         Softmax、交叉熵与三分类训练
│
├── 04-demos/                        两个入门 Demo
│   ├── demo01-numpy-linear-regression.py
│   ├── demo02-pytorch-classification.py
│   └── output/                      运行 Demo 01 后生成的图片
│
└── 05-large-language-model/         第四阶段：大语言模型基础
    ├── 01-token.md                  Token 与 Tokenizer
    ├── 02-embedding.md              Embedding 与位置信息
    ├── 03-transformer.md            Transformer 架构拆解
    ├── 04-attention.md              Attention 机制与因果掩码
    └── 05-llm-training.md           预训练、微调与对齐
```

各模块的具体描述见下一章第十一节。

---

## 十一、模块与文件说明

本章逐个说明仓库里每个文件的作用、关键结论与运行后应该看到的结果。
每个源文件开头也带有同主题的模块说明，代码与文档互为对照。

约定：

- 所有 `.py` 文件都是独立脚本，直接 `python3 路径/文件.py` 运行，不导入本项目其他文件。
- 第一阶段与第二阶段只依赖 NumPy；`04-demos/demo02` 需要 PyTorch。
- 建议顺序即文件编号顺序，不要跳过前四组直接看大模型部分。

---

### 11.1 01-mathematics：数学基础

这一组回答一个问题：**参数变化时，Loss 会怎么变。**
它是后面所有训练代码的数学前提，五个文件都不涉及模型训练，只练工具。

#### `01-function.py` — 函数

- **做什么**：把 `y = 2x + 1` 讲成映射，再说明模型 `y_pred = f(x)` 也是同一种东西。
- **关键内容**：固定输入 `x = [0, 1, 2, 3, 4]`，真实规律取 `y = 2x + 1`，
  然后依次换上 `(w=0,b=0)`、`(1,1)`、`(2,1)`、`(1.5,3)` 四组参数，
  打印每组预测值与平均绝对差距。
- **应看到**：只有 `w=2, b=1` 时差距为 0，其余组合都留下明显的平均差距。
- **要点**：训练不是在"改变规则的形式"，而是在"替换规则里的参数"。
  文件最后用一个两层模型说明层数变多之后，形式仍然是 `f(x)`。

#### `02-derivative.py` — 导数

- **做什么**：用差商 `(f(x+h) - f(x)) / h` 逼近导数，并与解析结果对照。
- **关键内容**：`y = x^2` 在 `x = 3` 处，`h` 从 `1e-1` 缩到 `1e-10`；
  `y = 2x + 1` 的导数处处为 2；再用 `Loss = (w - 2)^2` 表格说明导数符号的含义。
- **应看到**：`h` 减小到 `1e-5` 左右时数值导数非常接近 6；
  继续缩到 `1e-10`，误差反而变大，因为两个几乎相等的浮点数相减会丢有效位数。
  最后一张表里，`w < 2` 时导数为负（应当增大 `w`），`w = 2` 时导数为 0。
- **卡点提示**：导数是"局部"线性近似，只在足够小的邻域内成立，
  这正是后面必须用小步长更新参数的原因。

#### `03-gradient.py` — 梯度

- **做什么**：把单变量导数推广到多变量，并跑通一次真正的梯度下降。
- **关键内容**：目标函数 `L = w1^2 + 2 w2^2`，解析梯度 `[2w1, 4w2]`；
  用逐维中心差商验证；再比较沿梯度、负梯度、坐标轴、垂直梯度四个方向的方向导数。
- **应看到**：解析梯度与数值梯度一致；沿单位梯度方向方向导数最大，
  沿负梯度方向最小，垂直方向为 0；从 `(3, 3)` 出发、学习率 0.1，
  31 步后 Loss 降到 `1e-5` 量级。
- **延伸**：最后一节横向比较学习率 `0.01 / 0.1 / 0.4 / 0.6`，
  其中 `0.6` 会发散。原因是 `w2` 方向的二阶导数为 4，
  稳定更新要求学习率小于 `2 / 4 = 0.5`。

#### `04-chain-rule.py` — 链式法则

- **做什么**：证明"整条链的导数等于各段局部导数连乘"，为反向传播铺路。
- **关键内容**：`u = 2x + 1`、`y = u^2`，
  则 `dy/dx = dy/du * du/dx = 2u * 2 = 4(2x+1)`，与直接对复合函数数值求导对照；
  再算 `L = (w x + b)^2` 对 `w` 的梯度；最后摆出一个 `2 -> 3 -> 2` 的三层网络，
  打印前向传播的 `z1 / a1 / z2 / Loss`，并列出反向五步公式。
- **应看到**：两种求导方法数值完全一致；`x = 1` 时 `dy/dx = 12`。
- **要点**：每段局部导数只依赖本层的输入输出，因此可以复用缓存而不必重算整条链，
  这是反向传播比"逐个参数数值求导"便宜得多的根本原因。

#### `05-matrix.py` — 向量与矩阵

- **做什么**：建立形状意识，把线性层写成矩阵运算。
- **关键内容**：`(2,3) @ (3,) -> (2,)`；错误方向相乘会直接报错；
  批量写法 `Y = X @ W.T + b`，并与逐样本循环结果对比；
  参数量表 `Linear(5,5) = 25 + 5 = 30`；最后用三个词向量算内积与余弦相似度。
- **应看到**：批量计算与循环计算最大误差在 `1e-16` 量级；
  "苹果—香蕉"余弦相似度约 0.99，而与"汽车"只有约 0.2，
  说明内积可以衡量相似，这正是 Attention 用 `Q K^T` 的原因。
- **卡点提示**：写 PyTorch 时最常见的错误就是形状不匹配，
  建议在每一层后手动打印 `shape`，这个习惯本文件已在演示。

---

### 11.2 02-machine-learning：机器学习基础

这一组把数学工具第一次用到"数据"上，出现训练集、测试集与模型评估。

#### `01-linear-regression.py` — 线性回归与模型评估

- **做什么**：完整走一遍传统编程与机器学习的区别、数据划分、训练、评估。
- **关键内容**：数据仍由 `y = 2x + 3 + 噪声` 生成，但训练时不告诉模型这条规则；
  100 条数据打乱后前 80 训练、后 20 测试；
  梯度公式 `dw = mean(2 (y_pred - y) x)`、`db = mean(2 (y_pred - y))`，
  学习率 0.01，训练 1000 轮；评估使用测试集 MSE 与 `R^2`。
- **应看到**：`w` 收敛到约 2.02、`b` 约 2.81，
  测试集 MSE 约 0.5，`R^2` 约 0.986；
  训练集与测试集 Loss 同步下降，说明学到的是规律而不是记忆。
- **要点**：残差点来自噪声，属于不可预测部分，因此 Loss 不可能降到 0。
  这一节也是理解大模型 Loss 下限的类比。

#### `02-gradient-descent.py` — 梯度下降与学习率

- **做什么**：把"学习率决定成败"这件事变成可以直接看见的表格和字符画。
- **关键内容**：一维目标 `Loss = (w - 5)^2`，起点 `w = 0`；
  学习率对比 `0.001 / 0.05 / 0.1 / 0.5 / 1.1`；
  用 `ascii_curve` 打印三种学习率的收敛轨迹；
  最后比较批量、随机、小批量三种梯度下降。
- **应看到**：`0.5` 恰好一步落到最低点（更新式变成 `w - 1.0 * (w - 5) = 5`）；
  `1.1` 直接发散；三种梯度下降按"实际用过的样本数"对齐后，
  随机梯度下降用 6 万个样本达到与批量相近的 Loss，
  小批量只用 9600 个样本就接近收敛。
- **卡点提示**：比较优化方式时必须按数据使用量对齐，只比更新次数会得到误导性的结论，
  这也是代码里额外打印"用过样本数"这一列的原因。

---

### 11.3 03-neural-network：神经网络基础

这一组进入多神经元、多层结构，并手写完整的反向传播。
四个文件构成一条严格递进链：**结构 -> 前向 -> 反向 -> 分类任务**。

#### `01-linear-layer.py` — 线性层与激活函数

- **做什么**：手写 `Linear` 类（`z = x @ W.T + b`），实现四种激活函数，
  并用数值实验证明激活函数不可省略。
- **关键内容**：权重形状与 PyTorch `nn.Linear.weight` 一致，为 `(out, in)`；
  激活函数取值对照表覆盖 Sigmoid、Tanh、ReLU、GELU（GELU 用 tanh 近似式，避免依赖 SciPy）；
  把两层线性合并成一层 `W = W2 @ W1`、`b = W2 @ b1 + b2` 并比较输出。
- **应看到**：合并前后输出误差在 `1e-16`，即**没有激活函数时堆多少层都等价于一次线性变换**；
  加上 ReLU 后，一批 8 个神经元输出里有 5 个被截成 0。
- **要点**：ReLU 的稀疏输出既是优点（计算便宜、不易梯度消失），
  也埋下了"死神经元"的问题，这为反向传播那一节的梯度截断做了铺垫。

#### `02-forward-propagation.py` — 前向传播

- **做什么**：把一个 `3 -> 4 -> 2` 的网络逐层展开，打印每层形状与数值，并记录缓存。
- **关键内容**：`linear_forward` 返回 `z` 与 cache；
  单样本计算、批量计算（`(4,3)` 输入）两种走法；
  用 MSE 算出 Loss；最后给出与 PyTorch 写法的对应表。
- **应看到**：`z1/a1/z2` 形状依次为 `(4,4) / (4,4) / (4,2)`；
  手算与调用 `forward` 结果相同；16 个 `z1` 中有 7 个小于 0，
  这些位置反向传播时梯度会被 ReLU 截断。
- **要点**：cache 里同时保存了本层输入和输出，
  求 `dL/dW2` 要用 `a1`，求 `dL/dW1` 要用 `x` 和 `z1`，
  这解释了"为什么前向传播必须留中间结果"。

#### `03-backpropagation.py` — 反向传播

- **做什么**：不用任何框架，把上面那个网络的四个梯度完整推导、实现并验证。
- **关键内容**：
  `dL/dz2 = 2(z2 - target)/(n m)`、`dW2 = dz2.T @ a1`、`db2 = sum(dz2)`、
  `da1 = dz2 @ W2`、`dz1 = da1 * ReLU'(z1)`、`dW1 = dz1.T @ X`、`db1 = sum(dz1)`；
  随后用逐元素中心差商做梯度校验；再用自算梯度做 400 步梯度下降。
- **应看到**：解析梯度与数值梯度的最大相对误差在 `1e-9` 量级；
  梯度形状始终与参数形状一致；学习率 0.3 时 Loss 从 1.124 一路降到 `1e-5` 以下。
- **要点**：这是整个仓库最重要的一课。
  能独立写出并验证这七行公式，就意味着真正理解了 `loss.backward()`。
  最后对照表说明：优化器内部执行的正是 `W -= lr * grad`，只是加了动量等改进。

#### `04-classification.py` — Softmax 与交叉熵分类

- **做什么**：把训练闭环搬到分类任务上，手写 Softmax、交叉熵与对应梯度。
- **关键内容**：`softmax` 先减最大值防溢出；交叉熵表 `-log(0.9)=0.105` 到 `-log(0.1)=2.303`；
  数据是三个二维高斯簇，中心分别为 `(0,0)`、`(4,4)`、`(0,5)`，标准差 0.7；
  结构 `Linear(2,8) + ReLU + Linear(8,3)`，共 51 个参数；
  梯度用 Softmax 与交叉熵组合后的简洁形式 `dL/dlogits = (probs - one_hot) / n`。
- **应看到**：初始 Loss 约 1.75（高于瞎猜水平 `ln 3 = 1.099`，
  因为随机权重会自信地押注某一类），准确率仅 0.15；
  学习率 0.1、1500 轮后 Loss 约 0.017，测试准确率 1.0；
  查询点 `(0.3, 4.8)` 的概率约为 `[0.0002, 0.0013, 0.9985]`，正确落到簇 2。
- **要点**：把该文件学习率改成 1.0 会立刻看到 Loss 上下震荡、准确率不升反降，
  这是最直观的"学习率过大"现场演示。

---

### 11.4 04-demos：两个入门 Demo

#### `demo01-numpy-linear-regression.py`

纯 NumPy，对应第 4.3、4.4 节的公式实现：模型 `y_pred = w x + b`，
MSE 损失，手写梯度，学习率 0.01，1000 轮，
最终 `w` 约 2.016、`b` 约 2.810（真实值 2 和 3，残差来自噪声）。
`plot` 函数在 matplotlib 可用时既弹窗也保存图片到 `output/`，
未安装时只打印一行提示并继续完成训练，因此这个 Demo 在任何环境都能跑。

#### `demo02-pytorch-classification.py`

需要 PyTorch。任务是"输入五维向量，预测最大值所在下标"，
模型只有一个 `nn.Linear(5, 5)`，30 个参数，
训练集 5000、测试集 1000，Adam 学习率 0.001，50 个 epoch，小批量 32。
脚本开头会尝试导入 torch，缺失时打印安装命令并提示改看
`03-neural-network/04-classification.py`（同一套原理的纯 NumPy 版本）。

需要注意标签始终是 0～4，只有最后展示给用户时才 `+1`：
输入 `[0.1, 0.8, 0.3, 0.4, 0.2]` 时模型内部预测 1，输出显示 2。
这个任务用 `argmax` 一行就能解决，用神经网络只是为了练习训练流程。

---

### 11.5 05-large-language-model：大语言模型基础

这一组是概念文档，不要求运行环境，每篇都带可运行的 NumPy 片段。

| 文件 | 解决的问题 | 关键结论 |
|---|---|---|
| `01-token.md` | 模型怎么"看到"文字 | Token 是词表里的片段，长度不固定；ID 只是索引，没有大小含义；上下文长度按 Token 计 |
| `02-embedding.md` | 离散 ID 如何变成可计算的量 | Embedding 是形状 `(词表大小, d_model)` 的查表；语义来自训练中"相似上下文被推到一起" |
| `03-transformer.md` | 向量之间如何交换信息 | Attention 负责跨 Token，FFN 负责逐 Token 变换，残差让深层可训练，LayerNorm 稳定数值 |
| `04-attention.md` | 加权的权重从哪来 | `Softmax(QK^T / sqrt(d_k))V`；除以 `sqrt(d_k)` 是防 Softmax 饱和；因果掩码决定谁能看谁 |
| `05-llm-training.md` | 大模型到底在优化什么 | 目标仍是预测下一个 Token 的交叉熵；SFT 改变数据形态；对齐引入人类偏好；数学原理没有变 |

阅读这一组之前，务必确认已经理解 `03-neural-network/03-backpropagation.py`
和 `04-classification.py`：大模型训练循环与 Demo 02 的五行代码在结构上完全一致，
差别只在规模与工程实现。

---

## 十二、总结

学习神经网络，最重要的不是记住各种框架 API，而是理解模型训练背后的数学原理。

可以把整个学习过程归纳为：

```text
数学基础
   |
   v
理解函数和参数
   |
   v
理解导数和梯度
   |
   v
理解梯度下降
   |
   v
理解神经网络
   |
   v
理解前向传播和反向传播
   |
   v
使用 PyTorch 训练模型
   |
   v
理解 Transformer
   |
   v
理解大语言模型的学习过程
```

大语言模型看起来非常复杂，但它仍然建立在神经网络和数学优化的基础之上。

从一个简单的线性回归模型开始，逐步理解参数、Loss、梯度和反向传播，再去学习 Transformer 和大语言模型，会更容易建立完整的知识体系。

**学习原则：先理解数学原理，再通过代码验证，最后理解大模型如何将这些原理应用到海量数据和复杂网络中。**
