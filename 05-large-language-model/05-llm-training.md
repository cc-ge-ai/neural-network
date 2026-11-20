# 05 大模型是如何训练的：预训练、微调与对齐

> 对应详细文档：[docs/detailed-guide.md](../docs/detailed-guide.md) 第 6.1、6.6、6.7 节
> 前置知识：前面四篇 + [../03-neural-network/04-classification.py](../03-neural-network/04-classification.py)
> 本篇的 NumPy 片段可直接运行；PyTorch 片段需要安装 torch 后运行。

## 模块说明

这一篇把前面所有内容串起来：**大模型并没有使用新的数学，
它只是把"前向传播 -> 算 Loss -> 反向传播 -> 更新参数"这套流程，
在超大规模的数据、参数和算力上重复了极多次。**

读完本篇应能回答：

1. 预训练的监督信号从哪里来（为什么不需要人工标注）？
2. Loss 下降到大模型意义上意味着什么，困惑度怎么用？
3. 微调和对齐分别解决了预训练留下的什么问题？
4. 大模型训练与 Demo 02 的训练循环差在哪里？

---

## 一、训练目标只有一句话

**给模型大量文本，让它不断学习根据前面的内容预测下一个 Token。**

```text
输入：今天天气非常
候选：好 / 不错 / 热 / 差 ...
模型：为每个候选 Token 计算概率
真实：下一个 Token 是"好"
Loss：-log(P(好))
```

监督信号完全免费：任何一段文本，把它整体右移一位，
后一位就是前一位的答案。这就是语言建模能够使用互联网规模数据的原因，
也是它与分类任务最本质的差别——分类需要标签，语言建模只需要文本本身。

---

## 二、一段文本如何变成训练样本

```python
import numpy as np

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

# 一段已经 Tokenize 好的文本
ids = np.array([101, 2450, 87, 995, 1230, 67])

# 关键一步：右移一位，前一个位置预测后一个 Token
inputs = ids[:-1]
labels = ids[1:]
print("模型输入：", inputs)
print("预测目标：", labels)

# 真实模型这里会输出形状 (序列长度, 词表大小) 的 logits，先用随机数代替
# 注意词表大小必须大于样本里最大的 Token ID
rng = np.random.default_rng(0)
vocab_size = 3000
logits = rng.normal(size=(len(inputs), vocab_size))

probs = softmax(logits)
loss = -np.mean(np.log(probs[np.arange(len(inputs)), labels] + 1e-12))

print("交叉熵 Loss：", round(loss, 4))
print("瞎猜水平 ln(vocab_size) =", round(np.log(vocab_size), 4))
```

要点：

- 随机 logits 代表尚未训练的模型，它的 Loss 与瞎猜水平相当，这正是训练开始前的起点。
- 一次前向传播同时产生"每个位置预测下一个 Token"的监督，训练完全并行。
- 因果掩码保证位置 i 只能看到 `0..i`，否则答案会被直接看见。
- 真实训练会把长文本切成固定长度（如 2048、4096）的块，称为序列长度。

---

## 三、训练循环：与 Demo 02 结构完全一致

```python
import torch
import torch.nn as nn


def train_step(model, ids, optim):
    """ids 形状 (batch, seq_len)，目标是整段右移一位。"""
    inputs = ids[:, :-1]
    labels = ids[:, 1:]

    optim.zero_grad()

    logits = model(inputs)                       # (batch, seq_len - 1, vocab_size)

    # 交叉熵要求形状为 (样本数, 类别数) 与 (样本数,)
    logits = logits.reshape(-1, logits.size(-1))
    labels = labels.reshape(-1)

    loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()
    optim.step()

    return loss.item()
```

对照 Demo 02 与 03-neural-network 系列，五步一字未改：
`zero_grad -> forward -> loss -> backward -> step`。
差别只在模型规模、数据规模和工程实现上。

---

## 四、怎么判断模型学得好：Loss 与困惑度

语言模型最常用的指标是困惑度（Perplexity），它是交叉熵 Loss 的指数：

\[
PPL = e^{Loss}
\]

| Loss | 困惑度 | 含义 |
|---|---|---|
| 0.5 | 1.65 | 平均只在不到 2 个候选中犹豫 |
| 1.0 | 2.72 | 相当于在 3 个候选里挑 |
| 2.0 | 7.39 | 选择面明显变宽 |
| 3.0 | 20.09 | 预测仍然相当不确定 |
| 5.0 | 148.41 | 接近无结构文本的困难程度 |

直观理解：**困惑度是模型平均在多少个 Token 之间犹豫**。
词表大小为 V 时，完全瞎猜的 Loss 是 `ln(V)`，困惑度正好等于 V，
所以任何真实模型的困惑度都应远小于词表大小。

---

## 五、三个训练阶段

### 阶段一：预训练（Pre-training）

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

数据是网页、书籍、代码等海量原始文本，目标是学会语言的统计规律。
预训练结束后模型会"续写"，但并不擅长"回答问题"——
它学到的分布里，问句后面既可能跟答案，也可能跟另一个问句。

### 阶段二：监督微调（SFT）

使用人工编写或筛选的指令-回答数据，让模型学会按指令输出：

```json
{
  "instruction": "解释什么是神经网络",
  "output": "神经网络是一种由多个计算单元组成的机器学习模型..."
}
```

训练目标仍然只是预测下一个 Token，只是数据形态从"自然文本"变成
"一问一答"。一个常见细节是**只对回答部分计算 Loss**，指令部分被屏蔽：

```python
labels = torch.where(mask, next_ids, -100)     # -100 是 ignore_index
loss = nn.functional.cross_entropy(logits.view(-1, vocab), labels.view(-1), ignore_index=-100)
```

这样模型学习的是"如何作答"，而不是"如何复述问题"。

### 阶段三：偏好对齐（Alignment）

同一个问题可以有多个合理回答，SFT 数据只能给出其中一个。
对齐阶段用人类偏好告诉模型哪些回答更好：

- **RLHF**：先训练一个奖励模型学习人类对成对回答的偏好，
  再用强化学习让语言模型生成能获得更高奖励的回答。
- **DPO**：跳过奖励模型与强化学习，直接用成对偏好数据构造损失，
  让模型提高preferred回答的概率、降低 rejected 回答的概率。

不同模型的训练流程并不完全一致，不是所有模型都严格走满这三个阶段。

---

## 六、规模效应

同样的公式，把参数量、数据量、算力一起放大，会涌现出新能力。
经验上存在一条近似规律：损失主要由三者的规模共同决定，
任何一项单独放大而另两项跟不上，收益都会迅速饱和。

这也是为什么大模型研究的关键词从"设计新损失函数"变成
"数据配比、训练稳定性、显存与并行策略"：

- 混合精度：用更低位宽存储激活与权重，显存和吞吐大幅改善。
- 梯度累积：显存放不下大 batch 时，多步梯度累加后再更新一次。
- 分布式：张量并行、流水并行、数据并行组合起来，把一次训练摊到大量卡上。

数学没有变化，变化的只是工程规模。

---

## 七、与普通神经网络训练的对应关系

| 对比项目 | 普通神经网络 | 大语言模型 |
|---|---|---|
| 输入 | 数值、图片等 | Token ID / 向量 |
| 模型 | 线性层、CNN 等 | Transformer 等 |
| 输出 | 分类结果、预测数值 | Token 概率分布 |
| 损失函数 | MSE、交叉熵等 | 通常使用交叉熵 |
| 反向传播 | 计算参数梯度 | 计算参数梯度 |
| 优化器 | SGD、Adam 等 | AdamW 等常见优化器 |
| 训练目标 | 学习特定任务 | 学习语言规律及任务能力 |

一句话总结：**大模型没有脱离神经网络训练的基本原理，
它仍然通过前向传播、计算 Loss、反向传播和优化器更新参数来学习，
只是模型规模、训练数据和计算复杂度大幅提高。**

---

## 八、常见误解

| 误解 | 实际 |
|---|---|
| 预训练需要人工标注答案 | 标签来自文本自身，右移一位即可 |
| 模型像数据库一样存储事实 | 知识以参数中的统计关联形式存在，会出错也会编造 |
| 微调之后模型仍在实时学习 | 训练完成后参数固定，对话内容不会自动写入权重 |
| Loss 降到 0 才算学会 | 语言本身有不可消除的不确定性，Loss 不可能为 0 |
| 参数量越大一定越聪明 | 数据质量与规模同样关键，小模型好数据常胜过盲目堆参数 |

---

## 九、自测问题

1. 为什么"预测下一个 Token"这个目标可以免费获得海量训练数据？
2. 一个 Loss 为 2.0 的模型，困惑度是多少，它意味着什么？
3. SFT 阶段屏蔽指令部分的 Loss，会给模型带来什么不同？

回到简版说明：[../README.md](../README.md)；完整原理：[docs/detailed-guide.md](../docs/detailed-guide.md)。
