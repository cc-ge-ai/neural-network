# 02 Embedding：把 Token ID 变成向量

> 对应详细文档：[docs/detailed-guide.md](../docs/detailed-guide.md) 第 6.3 节
> 前置知识：[01-token.md](01-token.md)、[../01-mathematics/05-matrix.py](../01-mathematics/05-matrix.py)
> 本篇代码片段可直接保存为 `.py` 文件用 `python3` 运行（只依赖 NumPy）。

## 模块说明

上一节结束时，文本已经变成一串整数 ID。但整数之间没有可计算的语义关系：
`1024` 和 `1025` 相邻，不代表"苹果"和"香蕉"更接近。

Embedding 做的事就一件事：**用一个查表操作，把离散的 ID 换成连续的向量**，
从此文本进入了线性代数的世界，可以被加权、相加、算内积、求梯度。

读完本篇应能回答：

1. Embedding 矩阵的形状为什么是 `(词表大小, 向量维度)`？
2. 查表这一步在数学上等价于什么运算？
3. Embedding 里的语义是怎么来的，是谁训练出来的？
4. 模型的输出层和 Embedding 有什么关系？

---

## 一、查表：最直接的理解

```text
Token：    苹果
Token ID：  1024
Embedding：[0.12, 0.36, -0.25, 0.88, ...]
```

Embedding 层本质上是一个巨大的二维数组：

```python
embedding_table = np.zeros((vocab_size, d_model))   # 例如 (50257, 768)
vector = embedding_table[token_id]                  # 一次索引就完成"变换"
```

一批 Token 的 Embedding 就是把若干行取出来排在一起：

```python
ids = np.array([1024, 7, 55])          # 形状 (3,)
vectors = embedding_table[ids]         # 形状 (3, 768)
```

这正是 Transformer 输入的形状：**序列长度 × 向量维度**。

---

## 二、为什么不用 one-hot 向量

传统做法是给每个词一个长度为词表大小的向量，只有对应位置为 1：

| 表示方式 | 维度 | 缺点 |
|---|---|---|
| one-hot | 等于词表大小（几万到十几万） | 任意两个词的夹角都是 90 度，无法表达相似度 |
| Embedding | 固定 256 / 768 / 4096 | 维度小，且相似度可以从数值上算出来 |

更关键的是：one-hot 是人为规定的符号，而 Embedding 是**训练出来的**。
每个维度代表什么，事先没人设计过，是模型为了降低 Loss 自己形成的分布结构。

---

## 三、动手验证：查表 + 相似度

```python
import numpy as np

# 假想的一张极小 Embedding 表：5 个 Token，每个 4 维
# 真实模型的表是几万 × 几百，原理完全一样
table = np.array([
    [0.90, 0.10, 0.05, 0.00],   # 苹果
    [0.80, 0.20, 0.05, 0.00],   # 香蕉
    [0.10, 0.85, 0.05, 0.00],   # 汽车
    [0.05, 0.90, 0.05, 0.00],   # 自行车
    [0.30, 0.30, 0.30, 0.30],   # 今天
], dtype=float)

def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

names = ["苹果", "香蕉", "汽车", "自行车", "今天"]
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        print(f"{names[i]}-{names[j]}: {cosine(table[i], table[j]):.3f}")

# 查表：一句话 -> 向量矩阵
sentence_ids = np.array([0, 2, 4])     # 苹果 汽车 今天
print("\n句子向量形状:", table[sentence_ids].shape)
```

输出会显示：苹果与香蕉相似度很高，汽车与自行车很高，
而"水果"与"交通工具"两组之间相似度低。
**语义关系被编码进了向量几何关系里**，这就是 Embedding 的价值。

---

## 四、Embedding 是怎么"学会"这些含义的

模型训练目标只是预测下一个 Token。训练中每一步：

1. 输入 Token ID，取出 Embedding 向量。
2. 经过 Transformer 得到预测分布。
3. 与真实的下一个 Token 比较，算出交叉熵 Loss。
4. 梯度一路传回 Embedding 表，调整被查到的那些行。

由于梯度只更新本次被用到的行，一张几万行的表是分批、逐步被推动的。
最终形成的规律是：**出现在相似上下文里的 Token，向量会被推到一起**。
这不是人工设定的规则，而是降低 Loss 的自然结果。

---

## 五、位置信息也要加进来

Attention 本身不区分顺序（详见 [04-attention.md](04-attention.md)），
所以模型会把位置信息加到 Token Embedding 上：

```text
输入 = Token Embedding + Position Embedding
```

两者形状相同，直接逐元素相加。"苹果 吃 人" 与 "人 吃 苹果"
的 Token 集合一样，但位置不同，相加后向量也不同。

现代大模型常改用 RoPE 等相对位置方案，把位置直接作用在 Q 和 K 上，
但目的都一样：**让模型能感知顺序**。

---

## 六、输出层也常用同一张表

模型最后要把隐藏向量还原成词表上的概率分布：

```text
logits = hidden @ embedding_table.T     # 形状 (序列长度, 词表大小)
prob   = softmax(logits)
```

这一步常被称为 `lm_head`。很多模型会让输入 Embedding 与输出投影
**共享同一份权重**（weight tying），既省参数，又让"相似词"
在输入侧和输出侧保持一致的几何关系。

词表 50257、维度 768 时，仅这一张表就有约 3860 万个参数，
说明为什么词表大小不是越大越好。

---

## 七、常见误解

| 误解 | 实际 |
|---|---|
| Embedding 维度越多越好 | 维度越大参数越多，需要与算力平衡 |
| 每个维度有明确人类含义 | 维度是训练中自发形成的，通常不可解释 |
| Embedding 是预先设计好的词典 | 它是训练出来的参数，随训练不断被推动 |
| 一句一句话 Embedding 就是一个向量 | 模型内部每个 Token 位置都有一个向量 |
| 加了位置编码就不再需要顺序信息 | 位置编码正是为了补上 Attention 缺失的顺序信息 |

---

## 八、自测问题

1. 为什么 Embedding 查表可以用矩阵乘法等价表示？
2. 把词表从 5 万缩到 1 万，哪些部分的参数量会直接变化？
3. 如果去掉位置信息，`人 吃 苹果` 和 `苹果 吃 人` 在模型眼里有区别吗？

下一篇：[03-transformer.md](03-transformer.md) 讲这些向量如何被组织成一个大网络。
