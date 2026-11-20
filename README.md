# 神经网络与大语言模型学习指南

从数学基础出发，用简单的 Python 与 PyTorch 实验，逐步理解模型为什么能通过数据学习，
以及大语言模型是如何在同一套原理上放大规模得到的。

> 本文件是简明入口，只讲"怎么跑、先看什么"。
> 完整推导与每个模块的详细说明见 [docs/detailed-guide.md](docs/detailed-guide.md)。

---

## 快速开始

```bash
# 1. 安装依赖（前三个阶段只需要 numpy）
pip install -r requirements.txt

# 2. 从第一个文件开始按顺序运行
python 01-mathematics/01-function.py
python 01-mathematics/02-derivative.py

# 3. 两个完整 Demo
python 04-demos/demo01-numpy-linear-regression.py
python 04-demos/demo02-pytorch-classification.py
```

只需要 Python 3.9 以上。`01`、`02`、`03` 三个阶段全部只依赖 NumPy，
装好 NumPy 就能跑完大部分内容；只有 Demo 02 需要 PyTorch，
未安装时脚本会打印安装命令后退出，不影响其他文件。

---

## 项目结构

```text
neural-network/
├── README.md                    本文件
├── requirements.txt             依赖清单
├── docs/detailed-guide.md       详细版学习笔记 + 逐模块说明
│
├── 01-mathematics/              数学基础：函数、导数、梯度、链式法则、矩阵
├── 02-machine-learning/         机器学习基础：线性回归、梯度下降
├── 03-neural-network/           神经网络：线性层、前向、反向、分类
├── 04-demos/                    两个完整入门 Demo
└── 05-large-language-model/     大模型：Token、Embedding、Transformer、Attention、训练
```

学习路线严格递进：

```text
数学基础 -> 机器学习基础 -> 神经网络 -> 两个 Demo -> 大语言模型
```

---

## 模块索引

每个文件开头都带有一段模块说明，包含学习目标、运行方式与应该观察到的结论。

### 01-mathematics 数学基础

| 文件 | 一句话说明 |
|---|---|
| `01-function.py` | 函数是映射，模型就是带参数的映射 |
| `02-derivative.py` | 导数是变化率，用差商逼近并观察浮点精度限制 |
| `03-gradient.py` | 梯度由偏导数组成，指向 Loss 增长最快的方向 |
| `04-chain-rule.py` | 链式法则是反向传播的数学基础 |
| `05-matrix.py` | 线性层写成 `Y = X @ W.T + b`，建立形状意识 |

### 02-machine-learning 机器学习基础

| 文件 | 一句话说明 |
|---|---|
| `01-linear-regression.py` | 数据划分、训练、评估的最小完整闭环 |
| `02-gradient-descent.py` | 学习率过小变慢、过大发散，批量与随机梯度对比 |

### 03-neural-network 神经网络基础

| 文件 | 一句话说明 |
|---|---|
| `01-linear-layer.py` | 线性层与四种激活函数，证明没有激活函数时多层等于一层 |
| `02-forward-propagation.py` | 逐层前向传播，并缓存反向传播要用的中间结果 |
| `03-backpropagation.py` | 手算全部梯度，用数值差商校验到 1e-9 一致 |
| `04-classification.py` | Softmax 与交叉熵，训练三分类网络到接近满准确率 |

### 04-demos 入门 Demo

| 文件 | 一句话说明 |
|---|---|
| `demo01-numpy-linear-regression.py` | 不用框架，手动拟合 `y = 2x + 3`，绘图可选 |
| `demo02-pytorch-classification.py` | 用 PyTorch 训练五分类网络，预测最大值位置 |

### 05-large-language-model 大语言模型基础

| 文件 | 一句话说明 |
|---|---|
| `01-token.md` | Token 是模型处理文本的基本单位，与字数不等 |
| `02-embedding.md` | 把离散 ID 查表变成连续向量，语义来自训练 |
| `03-transformer.md` | Attention、FFN、残差、LayerNorm 各自解决什么问题 |
| `04-attention.md` | 逐步算一遍 `Softmax(QK^T / sqrt(d_k))V`，含因果掩码 |
| `05-llm-training.md` | 预训练、监督微调、偏好对齐三阶段与困惑度 |

---

## 建议学习顺序

| 顺序 | 内容 | 对应目录 |
|---|---|---|
| 1 | 函数、导数、梯度、链式法则 | `01-mathematics` |
| 2 | 线性回归与梯度下降、学习率 | `02-machine-learning` |
| 3 | 手动跑通一次梯度下降 | `04-demos/demo01` |
| 4 | 线性层、前向传播、反向传播、分类 | `03-neural-network` |
| 5 | PyTorch 训练流程 | `04-demos/demo02` |
| 6 | Token、Embedding、Attention、Transformer | `05-large-language-model` |
| 7 | 大模型训练三阶段 | `05-large-language-model/05-llm-training.md` |

不要一开始就研究大模型源码：前四个阶段的理解缺失，
会在阅读 Transformer 实现时被成倍放大。

---

## 一句话总结

学习神经网络，重要的不是记住框架 API，而是理解模型训练背后的数学原理：

```text
函数 -> 导数 -> 梯度 -> 梯度下降 -> 前向传播 -> 反向传播 -> 更新参数
```

大语言模型看起来复杂，但它仍然建立在同一套神经网络与数学优化之上，
只是模型规模、训练数据和计算复杂度大幅提高。
