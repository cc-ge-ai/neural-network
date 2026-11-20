"""
模块说明：向量与矩阵（Vector & Matrix）—— 神经网络的数据语言

对应详细文档：docs/detailed-guide.md 第 3.6 节
学习目标：
    1. 熟悉向量、矩阵的形状（shape）与矩阵乘法规则。
    2. 把线性层写成 y = Wx + b，并推广到整批数据 Y = XW^T + b。
    3. 会计算一个线性层的参数量，理解 nn.Linear(5, 5) 为什么是 30 个参数。
    4. 用内积衡量向量相似度，为后面理解 Attention 做准备。

运行：
    python 01-mathematics/05-matrix.py

观察要点：
    - 只有"前一个矩阵的列数 = 后一个矩阵的行数"才能相乘。
    - 一批数据只是把每个样本排成一行，层内部运算完全不变。
"""

import numpy as np


def section(title):
    print("\n" + "=" * 58)
    print(title)
    print("=" * 58)


def linear_params(in_features, out_features):
    """nn.Linear(in_features, out_features) 的参数量。"""
    return in_features * out_features + out_features


def main():
    np.set_printoptions(precision=3, suppress=True)

    section("一、向量与矩阵")

    x = np.array([0.1, 0.8, 0.3])
    print(f"向量 x = {x}，形状 {x.shape}：一组有序数字。")

    W = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"\n矩阵 W =\n{W}，形状 {W.shape}：二维数字表格，2 行 3 列。")

    print("\n矩阵乘法规则：(2x3) 乘 (3x1) -> (2x1)，结果每个元素是一行与一个向量的内积。")
    print(f"W @ x = {W @ x}")
    print(f"  第一项 = 1*0.1 + 2*0.8 + 3*0.3 = {1 * 0.1 + 2 * 0.8 + 3 * 0.3:.3f}")
    print(f"  第二项 = 4*0.1 + 5*0.8 + 6*0.3 = {4 * 0.1 + 5 * 0.8 + 6 * 0.3:.3f}")

    try:
        x @ W
    except ValueError as err:
        print(f"\n反过来用 (3,) 乘 (2,3) 会直接报错：{err}")

    section("二、线性层就是 y = Wx + b")

    b = np.array([0.5, -0.5])
    y = W @ x + b
    print(f"偏置 b = {b}")
    print(f"y = W @ x + b = {y}")
    print("其中 x 是输入向量，W 是权重矩阵，b 是偏置，y 是输出向量。")
    print(f"这一层的参数量 = 权重 {W.size} + 偏置 {b.size} = {W.size + b.size}")

    section("三、整批数据：Y = X W^T + b")

    X = np.array(
        [
            [0.1, 0.8, 0.3],
            [0.9, 0.2, 0.4],
            [0.5, 0.5, 0.5],
            [0.2, 0.1, 0.9],
        ]
    )
    print(f"一批 4 个样本排成矩阵 X，形状 {X.shape}（每行一个样本）")
    print(f"W 的形状 {W.shape}，即 (out_features, in_features)")
    print(f"X @ W.T 的形状 {X.dot(W.T).shape}，加偏置后广播到每一行")

    Y = X @ W.T + b
    print(f"Y =\n{Y}")

    print("\n对比：单个样本循环计算，结果与整批矩阵计算一致。")
    Y_loop = np.stack([W @ xi + b for xi in X])
    print(f"最大误差 = {np.max(np.abs(Y - Y_loop)):.3e}")
    print("框架内部用矩阵运算一次算完一批，这是 GPU 高效的原因。")

    section("四、维度与参数量")

    print(f"{'层定义':>22}{'权重':>10}{'偏置':>8}{'参数总数':>10}")
    for in_f, out_f in [(5, 5), (3, 128), (128, 128), (128, 4096)]:
        weights = in_f * out_f
        print(f"{f'Linear({in_f}, {out_f})':>22}{weights:>10}{out_f:>8}{weights + out_f:>10}")

    total = sum(linear_params(i, o) for i, o in [(4096, 4096), (4096, 4096)])
    print(f"\n两个 4096 x 4096 的线性层合计 {total:,} 个参数，已经接近小型 Transformer 一层的规模。")

    print("\n形状检查习惯（写 PyTorch 时会反复用到）：")
    print("  输入 (batch, in_features) @ 权重.T (in_features, out_features)")
    print("  -> 输出 (batch, out_features)")

    section("五、内积衡量相似度（Attention 的基础）")

    vectors = {
        "苹果": np.array([0.9, 0.1, 0.2]),
        "香蕉": np.array([0.8, 0.2, 0.1]),
        "汽车": np.array([0.0, 0.9, 0.7]),
    }
    print(f"{'词对':>12}{'内积':>10}{'余弦相似度':>14}")
    names = list(vectors)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, c = vectors[names[i]], vectors[names[j]]
            dot = a @ c
            cosine = dot / (np.linalg.norm(a) * np.linalg.norm(c))
            print(f"{names[i] + '-' + names[j]:>12}{dot:>10.3f}{cosine:>14.3f}")

    print("\n向量表示相近的事物时内积更大，这正是 Attention 用 Q 与 K 做内积的原因。")
    print("详见 05-large-language-model/04-attention.md。")


if __name__ == "__main__":
    main()
