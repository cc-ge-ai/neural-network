# 04 Attention：让模型按上下文取值

> 对应详细文档：[docs/detailed-guide.md](../docs/detailed-guide.md) 第 6.5 节
> 前置知识：[../01-mathematics/05-matrix.py](../01-mathematics/05-matrix.py)（内积与相似度）
> 本篇代码只依赖 NumPy，可保存为 `.py` 运行。

## 模块说明

Attention 是 Transformer 里唯一发生"跨 Token 信息交换"的地方。
它做的事情用一句话说就是：

**当前 Token 提出一个问题，序列里每个 Token 给出匹配度和内容，
模型按匹配度加权取回内容。**

读完本篇应能回答：

1. Q、K、V 三组向量分别从哪来、分别起什么作用？
2. 为什么要除以 `sqrt(d_k)`？
3. 因果掩码如何保证模型看不到未来的 Token？
4. 多头注意力为什么比单头更强？

---

## 一、先看一个语言现象

```text
小明把书放在桌子上，因为它很重。
```

"它"指什么？人、书，还是桌子？人类靠上下文瞬间判断，
模型需要一种机制，在处理"它"这个位置时，把前面几个 Token 的信息
按相关程度取回来。这个机制就是 Self-Attention。

---

## 二、核心公式

\[
Attention(Q, K, V) = Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

其中：

- Q：Query，查询向量，代表"我在找什么"。
- K：Key，键向量，代表"我能被什么匹配到"。
- V：Value，值向量，代表"匹配上之后实际取回的内容"。
- \(d_k\)：Key 向量的维度。

四步理解：

1. Q 和 K 做内积，算出两两关联分数。
2. Softmax 把每一行分数变成和为 1 的权重。
3. 用权重对 V 加权求和。
4. 得到融合了上下文信息的新向量。

Q、K、V 都不是额外标注出来的，而是同一个输入矩阵乘三组不同的可学习权重：

```text
Q = X @ Wq
K = X @ Wk
V = X @ Wv
```

训练会把这些权重调整成"对预测下一个 Token 最有帮助"的样子。

---

## 三、动手算一遍

```python
import numpy as np

rng = np.random.default_rng(0)

# 三个 Token，每个四维：小明 / 书 / 它
X = np.array([
    [0.9, 0.1, 0.2, 0.0],
    [0.3, 0.8, 0.1, 0.2],
    [0.2, 0.9, 0.3, 0.1],
])

Wq, Wk, Wv = (rng.normal(0, 0.5, (4, 4)) for _ in range(3))
Q, K, V = X @ Wq.T, X @ Wk.T, X @ Wv.T

d_k = K.shape[-1]

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

scores = Q @ K.T / np.sqrt(d_k)
weights = softmax(scores)

print("关联分数:\n", np.round(scores, 3))
print("注意力权重（每行之和为 1）:\n", np.round(weights, 3))
print("行和:", np.round(weights.sum(axis=1), 6))

output = weights @ V
print("加权取值后的新向量:\n", np.round(output, 3))
```

关注输出里的两点：

- `weights` 是一个 `3 x 3` 矩阵，第 i 行第 j 列表示"第 i 个 Token
  在第 j 个 Token 上投入了多少注意力"，每行和为 1。
- `output` 的每一行都是全部 V 的加权平均，
  因此每个位置的表示都已经掺入了上下文信息。

---

## 四、为什么要除以 sqrt(d_k)

Q 与 K 的分量近似独立、方差约 1 时，`d_k` 个乘积相加的内积方差约为 `d_k`，
标准差约 `sqrt(d_k)`。维度越高，分数之间的差距越容易被拉大，
Softmax 会把权重压成近似 one-hot，梯度随之消失。

除以 `sqrt(d_k)` 把分数尺度拉回方差 1 的量级，让 Softmax 保持在
"有偏好但不极端"的区间。这是一个纯数值稳定技巧，却不加就会训练困难。

---

## 五、因果掩码：不许偷看答案

语言模型训练时整段文本同时输入，但位置 t 的预测只能依赖前 t-1 个 Token。
做法是在 Softmax 之前把"未来"的分数设为负无穷：

```python
# 接上一段已定义的 scores、softmax
scores = Q @ K.T / np.sqrt(d_k)

# 上三角（不含对角线）标记为需要屏蔽的未来位置
mask = np.triu(np.ones_like(scores, dtype=bool), k=1)
scores = np.where(mask, -np.inf, scores)

weights = softmax(scores)
print(np.round(weights, 3))
```

第一行只有自己能看见，所以权重全部分给自己；
第三行可以看到前三者。这样一次前向传播就能同时训练全部位置，
而不必真的把序列逐个截断跑 N 遍。

对应关系一句话概括：**掩码决定了"谁能看谁"，
这也正是训练能并行、生成必须串行的根本原因。**

---

## 六、多头注意力

把 `d_model` 切成若干个 `head_dim`，每个头独立做 Attention，再把结果拼接：

\[
head\_dim = d\_model / num\_heads
\]

```python
import numpy as np

seq_len, d_model, num_heads = 512, 768, 12
head_dim = d_model // num_heads          # 64

# Q 形状 (序列长度, 768)，拆成 (序列长度, 12, 64)
Q = np.random.default_rng(0).normal(size=(seq_len, d_model))
Q_heads = Q.reshape(seq_len, num_heads, head_dim)

print(Q.shape, "->", Q_heads.shape)
print("每个头独立做一次 Attention，处理的维度只有", head_dim)
```

单头只能表达一种关系分布；多头可以同时进行多种"关注"：
有的头跟踪邻近词，有的头跟踪长距离指代，有的头关注句法主干。
拼接后再用一个输出投影 `Wo` 混合各头结果。

参数量并没有因为多头而增加：
每个头只是同一个大矩阵的一块切片，总维度仍是 `d_model x d_model`。

---

## 七、代价：平方级的复杂度

注意力分数矩阵的形状是 `(n, n)`，n 为序列长度：

| 序列长度 | 分数矩阵元素数 | 相对成本 |
|---|---|---|
| 512 | 262 144 | 1x |
| 4 096 | 16 777 216 | 64x |
| 32 768 | 1 073 741 824 | 4096x |

长度翻 8 倍，注意力计算量翻 64 倍。因此长上下文真正昂贵，
工程上发展出 FlashAttention（优化显存读写）、
稀疏或分块注意力、GQA（分组共享 KV）等手段来缓解。

推理阶段还要缓存历史 Token 的 K 和 V（KV Cache），
避免每生成一个新 Token 都重算整段，代价是显存随序列长度线性增长。

---

## 八、常见误解

| 误解 | 实际 |
|---|---|
| Q、K、V 是三份不同的输入 | 都来自同一个 X，只是乘了三组不同权重 |
| Attention 会记住整段原文 | 它输出的只是加权后的向量，不是检索数据库 |
| 头数越多一定越好 | 头数增加会让每头维度变小，需要与 `d_model` 平衡 |
| 掩码只影响推理 | 训练时同样必须掩码，否则答案会被直接看见 |
| 注意力权重就是模型的"理解" | 权重可解释性有限，某些头并不对应人类概念 |

---

## 九、自测问题

1. 若把 `d_k` 从 64 增到 256，不缩放会带来什么问题？
2. 为什么一次带掩码的前向传播能同时训练序列中每个位置的预测？
3. 多头之间如何拼接？总参数量为什么没有变化？

下一篇：[05-llm-training.md](05-llm-training.md) 讲这套架构如何被训练成大语言模型。
