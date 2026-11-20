# 03 Transformer：现代大模型的骨架

> 对应详细文档：[docs/detailed-guide.md](../docs/detailed-guide.md) 第 6.4 节
> 前置知识：[../01-mathematics/05-matrix.py](../01-mathematics/05-matrix.py)、
> [02-embedding.md](02-embedding.md)、[04-attention.md](04-attention.md)

## 模块说明

Embedding 给出了每个 Token 的向量，但这些向量此刻还互相独立。
Transformer 负责的是下一步：**让每个 Token 都能参考上下文里的其他 Token，
并把这个信息融合进自己的表示**。

本篇按组件拆解整个架构，重点说明"每个组件解决了什么单独无法解决的问题"。

读完本篇应能回答：

1. 为什么不能只用一串线性层来处理文本？
2. 一个 Transformer Block 里六个部件各自的作用是什么？
3. 残差连接为什么是深层网络的必需品？
4. 大模型为什么大多是 Decoder-only 结构？

---

## 一、Transformer 之前的困境

处理序列最早的做法是循环网络：从左到右逐字读，把历史信息压进一个隐藏状态。
两个问题让它无法支撑大模型：

- **无法并行**：第 100 个词必须等前 99 个词算完，训练海量数据时算力被串行卡死。
- **长距离信息衰减**：句子开头的内容要经过上百次传递才能到达句尾，很容易被稀释掉。

Transformer 的做法是：**一次性让所有位置互相看见**，用矩阵运算并行完成，
任何两个 Token 之间的信息距离都是一步。

---

## 二、整体结构

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
Transformer Block（重复 N 次）
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

数据在 Block 里始终是这个形状：

```text
(batch, 序列长度, d_model)
```

例如 `(32, 1024, 768)`。层数增加，形状不变，只是内部表示被反复精炼。

---

## 三、逐个组件拆解

### 3.1 Multi-Head Attention：让 Token 之间互相取经

唯一发生"跨 Token 信息交换"的地方。每个位置生成 Query，与所有位置的 Key
算相关性，再按相关性加权取回 Value。详细推导见 [04-attention.md](04-attention.md)。

多头则是把 `d_model` 切成若干份并行做注意力，让模型同时关注不同类型的关系
（语法、指代、距离等）。

### 3.2 Feed Forward Network：逐位置的非线性变换

```text
FFN(x) = W2 @ act(W1 @ x + b1) + b2
```

对序列里每个位置独立执行，通常是 `d_model -> 4 * d_model -> d_model`
的两层线性加激活（激活常用 GELU）。

一个直观分工：Attention 负责" mixing across tokens"，
FFN 负责"transforming each token"。模型记住的大量事实性关联，
经验上主要存储在这些 FFN 权重里。

### 3.3 Residual Connection：把结果加回输入

```text
x = x + Attention(LayerNorm(x))
```

作用有两个：

1. 提供一条"直通"路径，即使这一层什么也没学到，信息也不会丢。
2. 反向传播时梯度可以沿残差直接回传，几十个 Block 连起来仍能被训练。

没有残差连接，几十层深度的网络几乎无法优化，这是深度学习能"深"起来的关键工程之一。

### 3.4 Layer Normalization：稳定每一层的数值分布

对每个位置的 `d_model` 维向量做均值方差归一化，再仿射变换。
数值稳定后可以使用更大的学习率，训练不容易发散。
现代模型常把它放到子层之前（Pre-LN），深层训练更平稳。

---

## 四、一个 Block 的 PyTorch 写法

下面这段是简化实现，结构与真实模型一致，省略了因果掩码的细节：

```python
import torch
import torch.nn as nn


class Block(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        # 残差分支一：注意力负责跨 Token 交换信息
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # 残差分支二：前馈网络负责逐位置变换
        x = x + self.ffn(self.norm2(x))
        return x
```

模型主体只是把它重复 N 次，最后接一个输出投影：

```python
model = nn.Sequential(*[Block(768, 12) for _ in range(12)], nn.LayerNorm(768))
logits = model(embedding)          # (batch, seq, d_model) -> 再投影到词表
```

---

## 五、参数量都花在哪里

以 `d_model = 768`、12 层为例，单个 Block 的量级：

| 组件 | 参数构成 | 大约数量 |
|---|---|---|
| Attention 的 Q/K/V/O | 4 个 `768 x 768` | 236 万 |
| FFN | `768 x 3072` + `3072 x 768` | 472 万 |
| 两个 LayerNorm | `2 x 768 x 2` | 约 3 千 |

FFN 约占总参数的三分之二。把 `d_model` 翻倍，Block 参数量约变成四倍。
再算上 Embedding（词表 5 万 × 768 约 3860 万），
就能理解"参数量"这个数字是怎么被架构选择决定的。

---

## 六、Encoder、Decoder 与 Decoder-only

| 结构 | Attention 范围 | 典型用途 |
|---|---|---|
| Encoder | 双向，每个词能看左右两侧 | 理解类任务、翻译输入侧 |
| Decoder | 单向，只能看左侧（因果掩码） | 生成类任务、翻译输出侧 |
| Decoder-only | 单向，堆叠很多 Block | 当今绝大多数大语言模型 |

大模型选择 Decoder-only 的原因很直接：
训练目标是"根据前文预测下一个 Token"，这本身就是单向任务；
结构统一之后，同一套模型既能理解也能生成，还最容易扩展到海量数据。

---

## 七、生成文本时发生什么

```text
已有序列 -> 前向传播 -> 最后一个位置的下一个 Token 分布
         -> 采样得到一个 Token -> 拼回序列 -> 再来一次
```

```python
def generate(model, ids, max_new_tokens):
    for _ in range(max_new_tokens):
        logits = model(ids)              # 只需要最后一个位置的 logits
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids
```

这就是自回归生成：**每生成一个字，都要把整段已生成内容再喂进去一次**。
序列越长，注意力计算越贵，这也是上下文长度会显著影响推理成本的原因。

---

## 八、常见误解

| 误解 | 实际 |
|---|---|
| Transformer 靠层数理解语法 | 语法关系体现在 Attention 学到的权重分布上 |
| 层数越多越聪明 | 更深要配套更多数据和更稳的优化设置，否则收益递减 |
| Attention 是唯一的计算部件 | FFN 参数量更大，承担了大量变换 |
| 残差连接只是加速收敛 | 它同时是梯度能传回浅层的前提 |
| 训练和生成用的是同一套计算 | 生成时可以复用 KV 缓存，训练可以整段并行 |

---

## 九、自测问题

1. 去掉残差连接，为什么深层 Transformer 很难训练？
2. Attention 与 FFN 分别在"跨 Token"和"逐 Token"上做什么？
3. 为什么 Decoder-only 结构适合做语言生成？

配套阅读：[04-attention.md](04-attention.md) 拆解其中最关键的注意力机制。
