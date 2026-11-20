"""
模块说明：线性层与激活函数（Linear Layer & Activation）

对应详细文档：docs/detailed-guide.md 第 5.1、5.2 节
学习目标：
    1. 手写一个线性层，理解 z = Wx + b 中每个部分的形状与含义。
    2. 掌握 Sigmoid、Tanh、ReLU、GELU 四种激活函数的取值特性。
    3. 用数值实验证明：没有激活函数，多层网络会退化成一个线性层。
    4. 理解 PyTorch 中 nn.Linear(5, 5) 为什么是 30 个参数。

运行：
    python 03-neural-network/01-linear-layer.py

观察要点：
    - 权重矩阵形状为 (输出维度, 输入维度)，与 nn.Linear 的 weight 一致。
    - 两层线性直接相连，等价于一层线性。
    - ReLU 把负数截成 0，因此一半神经元在某次前向传播中可能不输出任何信息。
"""

import numpy as np


class Linear:
    """最小线性层：z = x @ W.T + b。

    参数
        in_features:  输入向量长度
        out_features: 输出向量长度
    """

    def __init__(self, in_features, out_features, seed=0):
        rng = np.random.default_rng(seed)

        # 与 PyTorch 的 nn.Linear 一致：weight 形状为 (out_features, in_features)
        # 用小随机数初始化，避免所有神经元学到完全相同的东西
        self.W = rng.normal(0, 0.5, size=(out_features, in_features))
        self.b = np.zeros(out_features)

    def __call__(self, x):
        return x @ self.W.T + self.b

    @property
    def num_params(self):
        return self.W.size + self.b.size


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def tanh(z):
    return np.tanh(z)


def relu(z):
    return np.maximum(0, z)


def gelu(z):
    """GELU 的 tanh 近似式，Transformer 中的 Feed Forward 常用它。"""
    return 0.5 * z * (1 + tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    np.set_printoptions(precision=3, suppress=True)

    section("一、一个神经元就是一个线性层加一个激活函数")

    layer = Linear(in_features=2, out_features=1, seed=1)
    x = np.array([0.6, 0.2])

    z = layer(x)
    a = sigmoid(z)
    print(f"输入 x = {x}")
    print(f"权重 W = {layer.W}，偏置 b = {layer.b}")
    print(f"z = Wx + b = {z}          （加权求和）")
    print(f"a = sigmoid(z) = {a}     （激活函数）")
    print("单个神经元的计算过程正是：先线性组合，再非线性映射。")

    section("二、一层里有多个神经元，就是矩阵运算")

    layer = Linear(in_features=3, out_features=4, seed=2)
    x = np.array([0.1, 0.8, 0.3])
    z = layer(x)
    print(f"输入形状 {x.shape} -> 权重形状 {layer.W.shape} -> 输出形状 {z.shape}")
    print(f"z = {z}")
    print(f"这一层参数量 = 权重 {layer.W.size} + 偏置 {layer.b.size} = {layer.num_params}")

    print(f"\n{'层定义':>20}{'权重':>8}{'偏置':>8}{'总参数':>10}")
    for in_f, out_f in [(5, 5), (4, 128), (128, 512)]:
        layer = Linear(in_f, out_f)
        print(f"{f'Linear({in_f}, {out_f})':>20}{layer.W.size:>8}{layer.b.size:>8}{layer.num_params:>10}")

    print("\nREADME 里 Demo 02 用的是 nn.Linear(5, 5)，即 25 个权重加 5 个偏置，共 30 个参数。")

    section("三、常见激活函数取值对照")

    z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    print(f"{'z':>8}{'Sigmoid':>12}{'Tanh':>10}{'ReLU':>10}{'GELU':>10}")
    for zi in z:
        print(
            f"{zi:>8.1f}{sigmoid(zi):>12.4f}{tanh(zi):>10.4f}"
            f"{relu(zi):>10.4f}{gelu(zi):>10.4f}"
        )

    print("\n特性小结：")
    print("  Sigmoid 把输出压到 0~1，可当作概率，但两端梯度接近 0。")
    print("  Tanh 把输出压到 -1~1，零附近变化最快。")
    print("  ReLU 负数归零、正数原样通过，计算便宜且不容易梯度消失。")
    print("  GELU 是 ReLU 的平滑版本，Transformer 里更常见。")

    section("四、为什么必须有激活函数")

    in_dim, hidden, out_dim = 3, 4, 2
    L1 = Linear(in_dim, hidden, seed=3)
    L2 = Linear(hidden, out_dim, seed=4)

    x = np.array([[0.1, 0.8, 0.3], [0.9, 0.2, 0.4]])

    # 两层线性直接相连，没有任何激活函数
    two_layer = L2(L1(x))

    # 合并成一个线性层：W = W2 @ W1，b = W2 @ b1 + b2
    merged = Linear(in_dim, out_dim)
    merged.W = L2.W @ L1.W
    merged.b = L2.W @ L1.b + L2.b
    one_layer = x @ merged.W.T + merged.b

    print(f"两层线性输出:\n{two_layer}")
    print(f"合并为一层线性输出:\n{one_layer}")
    print(f"两者最大误差 = {np.max(np.abs(two_layer - one_layer)):.3e}")
    print("\n结论：没有激活函数时，无论堆多少层都等价于一次线性变换，")
    print("      模型只能拟合直线或平面，无法学习复杂规律。")

    section("五、加上激活函数后就不一样了")

    hidden_out = relu(L1(x))
    print(f"隐藏层经过 ReLU:\n{hidden_out}")
    print(f"输出层结果:\n{relu(L1(x)) @ L2.W.T + L2.b}")
    print("每加一层线性 + 激活，模型就多掌握一段可以学习的非线性变换。")

    zero_count = int(np.sum(relu(L1(x)) == 0))
    print(f"\n这批数据一共有 {hidden_out.size} 个神经元输出，其中 {zero_count} 个被 ReLU 截成 0。")


if __name__ == "__main__":
    main()
