"""
模块说明：前向传播（Forward Propagation）—— 数据如何一路算到 Loss

对应详细文档：docs/detailed-guide.md 第 5.3 节第一步
学习目标：
    1. 逐层算出 z 与 a，并记录每一层的形状与数值。
    2. 理解前向传播必须缓存中间结果，供反向传播复用。
    3. 从预测结果算出 Loss，完成训练闭环的前半段。
    4. 对齐 PyTorch 中 model(x) 与损失函数这两行代码的含义。

运行：
    python 03-neural-network/02-forward-propagation.py

观察要点：
    - 网络结构：3 个输入 -> 4 个隐藏神经元 -> 2 个输出。
    - 一批数据只是把每个样本排成一行，层内运算完全不变。
    - cache 里留下的 z1、a1 正是下一个模块求梯度时要用的东西。
"""

import numpy as np


def relu(z):
    return np.maximum(0, z)


def linear_forward(x, W, b):
    """一层线性变换：z = x @ W.T + b，同时保存反向传播需要的输入。"""
    z = x @ W.T + b
    cache = {"x": x, "W": W, "b": b, "z": z}
    return z, cache


def forward(model, x):
    """完整前向传播：线性层 -> ReLU -> 线性层，返回输出和全部缓存。"""
    cache = {}

    z1, c1 = linear_forward(x, *model["W1b1"])
    a1 = relu(z1)

    z2, c2 = linear_forward(a1, *model["W2b2"])

    # 分类任务这里就是 logits；本模块先用回归视角，直接和 target 比较
    cache["layer1"] = {"x": x, "z1": z1, "a1": a1, "W1": c1["W"], "b1": c1["b"]}
    cache["layer2"] = {"a1": a1, "z2": z2, "W2": c2["W"], "b2": c2["b"]}

    return z2, cache


def mse_loss(y_pred, target):
    """均方误差，对批次内所有元素取平均。"""
    return np.mean((y_pred - target) ** 2)


def init_model(seed=0):
    """按 (输出维度, 输入维度) 初始化两层网络的权重与偏置。"""
    rng = np.random.default_rng(seed)
    return {
        "W1b1": (rng.normal(0, 0.5, size=(4, 3)), np.zeros(4)),
        "W2b2": (rng.normal(0, 0.5, size=(2, 4)), np.zeros(2)),
    }


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    np.set_printoptions(precision=3, suppress=True)

    section("一、网络结构")

    model = init_model(seed=0)
    W1, b1 = model["W1b1"]
    W2, b2 = model["W2b2"]

    print("输入 3 维 -> 隐藏层 4 个神经元（ReLU）-> 输出 2 维")
    print(f"W1 形状 {W1.shape}，b1 形状 {b1.shape}，参数量 {W1.size + b1.size}")
    print(f"W2 形状 {W2.shape}，b2 形状 {b2.shape}，参数量 {W2.size + b2.size}")
    print(f"整个网络参数总量 = {W1.size + b1.size + W2.size + b2.size}")

    section("二、单个样本逐层计算")

    x = np.array([0.1, 0.8, 0.3])
    print(f"输入 x = {x}，形状 {x.shape}")

    z1 = x @ W1.T + b1
    a1 = relu(z1)
    print(f"\n第一层 z1 = W1 x + b1 = {z1}，形状 {z1.shape}")
    print(f"        ReLU 后 a1    = {a1}")

    z2 = a1 @ W2.T + b2
    print(f"第二层 z2 = W2 a1 + b2 = {z2}，形状 {z2.shape}")
    print("输出 z2 就是模型的预测结果（未归一化的分数，PyTorch 里叫 logits）。")

    print("\n逐层拆开写和调用 forward 结果一致：")
    y, _ = forward(model, x.reshape(1, -1))
    print(f"forward 输出 = {y[0]}，与手算 {z2} 相同")

    section("三、一批样本同时计算")

    X = np.array(
        [
            [0.1, 0.8, 0.3],
            [0.9, 0.2, 0.4],
            [0.5, 0.5, 0.5],
            [0.2, 0.1, 0.9],
        ]
    )
    y_pred, cache = forward(model, X)

    print(f"批次输入形状 {X.shape}（4 个样本，每个 3 维）")
    print(f"第一层 z1 形状 {cache['layer1']['z1'].shape}")
    print(f"第一层 a1 形状 {cache['layer1']['a1'].shape}")
    print(f"输出形状 {y_pred.shape}")
    print(f"\n每个样本的预测：\n{y_pred}")

    section("四、计算 Loss")

    target = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    loss = mse_loss(y_pred, target)
    print(f"目标：\n{target}")
    print(f"Loss = mean((y_pred - target)^2) = {loss:.6f}")
    print("此时参数还是随机初始化的，Loss 完全由初始权重决定，没有任何'学习'。")

    section("五、前向传播到底缓存了什么")

    for name, layer in cache.items():
        shapes = {k: v.shape for k, v in layer.items()}
        print(f"{name}: {shapes}")

    print("\n缓存里既有本层输入，也有本层输出，反向传播时不需要重算：")
    print("  求 dL/dW2 需要 a1；求 dL/dW1 需要 x 和 ReLU 的输入 z1。")

    relu_input_negative = int(np.sum(cache["layer1"]["z1"] < 0))
    print(f"\n本批 {cache['layer1']['z1'].size} 个 z1 中有 {relu_input_negative} 个小于 0，")
    print("这些位置 ReLU 导数为 0，反向传播时梯度会被截断，传不到前面的层。")

    section("六、与 PyTorch 代码的对应关系")

    print("本地写法  ->  PyTorch 等价写法")
    for local, torch_code in [
        ("y_pred = forward(model, x)", "y_pred = model(x)"),
        ("z = x @ W.T + b", "nn.Linear(in, out)"),
        ("a1 = relu(z1)", "nn.ReLU()"),
        ("loss = mse_loss(y_pred, target)", "nn.functional.mse_loss(...)"),
        ("loss = cross_entropy(logits, y)", "nn.functional.cross_entropy(...)"),
    ]:
        print(f"  {local}  ->  {torch_code}")

    print("\n下一步：03-neural-network/03-backpropagation.py 用这里的 cache 手算梯度。")


if __name__ == "__main__":
    main()
