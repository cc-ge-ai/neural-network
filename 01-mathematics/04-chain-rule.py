"""
模块说明：链式法则（Chain Rule）—— 反向传播的数学基础

对应详细文档：docs/detailed-guide.md 第 3.5 节
学习目标：
    1. 理解复合函数求导：dy/dx = dy/du * du/dx。
    2. 会用"每层只看自己"的局部导数拼出整条链的导数。
    3. 理解多层网络中，最终 Loss 对前面某一层参数的梯度如何一段一段传回去。
    4. 明白反向传播就是链式法则 + 中间结果复用。

运行：
    python 01-mathematics/04-chain-rule.py

观察要点：
    - 链式法则结果与直接对复合函数数值求导完全一致。
    - 每段局部导数只依赖本层的输入输出，因此可以逐层复用。
"""

import numpy as np


def section(title):
    print("\n" + "=" * 58)
    print(title)
    print("=" * 58)


def central_difference(f, x, eps=1e-5):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def main():
    section("一、一个两层复合函数")

    print("设 u = 2x + 1，y = u^2，求 dy/dx。")
    print("直接代入：y = (2x + 1)^2。")
    print("链式法则：dy/dx = dy/du * du/dx = 2u * 2 = 4(2x + 1)。")

    def u_of_x(x):
        return 2 * x + 1

    def y_of_u(u):
        return u**2

    def y_of_x(x):
        return y_of_u(u_of_x(x))

    print(f"\n{'x':>6}{'u':>8}{'y':>10}{'链式法则':>12}{'数值求导':>12}")
    for x in [0.0, 1.0, 2.0, 3.0, -0.5]:
        u = u_of_x(x)

        # 局部导数：y 对 u 求导，u 对 x 求导
        dy_du = 2 * u
        du_dx = 2.0

        chain = dy_du * du_dx
        numerical = central_difference(y_of_x, x)
        print(f"{x:>6.1f}{u:>8.1f}{y_of_x(x):>10.2f}{chain:>12.4f}{numerical:>12.4f}")

    print("\n两种做法一致：整条链的导数 = 各段局部导数连乘。")

    section("二、为什么要拆成链")

    print("单看一层，导数都非常简单：")
    print("  u = 2x + 1   ->  du/dx = 2")
    print("  y = u^2      ->  dy/du = 2u")
    print("网络再深也一样，每个节点只需要知道自己对上一个节点的局部导数。")

    print("\n三层链：L = (w * x + b)^2，拆成 a = w*x + b，L = a^2")
    x, w, b = 2.0, 3.0, 1.0

    a = w * x + b
    L = a**2

    dL_da = 2 * a          # 第三段
    da_dw = x              # 第二段（a 对 w 的局部导数）

    dL_dw = dL_da * da_dw
    print(f"  x = {x}, w = {w}, b = {b}")
    print(f"  a = w*x + b = {a}")
    print(f"  L = a^2 = {L}")
    print(f"  dL/da = {dL_da}，da/dw = {da_dw}")
    print(f"  dL/dw = dL/da * da/dw = {dL_dw}")

    def L_of_w(w_):
        return (w_ * x + b) ** 2

    print(f"  数值校验 dL/dw = {central_difference(L_of_w, w):.4f}")

    section("三、反向传播 = 链式法则 + 复用中间结果")

    print("前向传播记录的中间量：")

    # 模拟一个三层网络：x -> z1 -> h1 -> z2 -> Loss
    # 权重按 x @ W 的写法排布，形状为 (输入维度, 输出维度)
    x_vec = np.array([1.0, 2.0])
    W1 = np.array([[0.5, 0.1, 0.3], [-0.2, 0.8, 0.4]])    # 2 -> 3
    b1 = np.array([0.1, 0.0, -0.1])
    W2 = np.array([[0.6, -0.3], [0.1, 0.4], [0.2, -0.5]])  # 3 -> 2
    b2 = np.array([0.0, 0.2])
    target = np.array([1.0, 0.0])

    z1 = x_vec @ W1 + b1
    h1 = np.maximum(z1, 0)          # ReLU
    z2 = h1 @ W2 + b2
    diff = z2 - target
    L = np.mean(diff**2)

    print(f"  z1 = {z1}")
    print(f"  h1 = ReLU(z1) = {h1}")
    print(f"  z2 = {z2}")
    print(f"  Loss = {L:.6f}")

    print("\n从 Loss 往回逐段相乘（只写标量形式，向量形式见 03-neural-network/03-backpropagation.py）：")
    print("  dL/dz2 = 2 * (z2 - target) / n              -> 本层局部导数")
    print("  dL/dW2 = h1^T @ dL/dz2                       -> 再乘 h1 对 W2 的导数")
    print("  dL/dh1 = dL/dz2 @ W2^T                        -> 继续往上传")
    print("  dL/dz1 = dL/dh1 * ReLU'(z1)                   -> 再乘激活函数导数")
    print("  dL/dW1 = x^T @ dL/dz1")

    print("\n关键点：")
    print("  1. 每一段只需要前向传播缓存下来的输入，不必重新计算整条链。")
    print("  2. 同一段导数被多个后续计算共用，所以比逐个参数数值求导便宜得多。")
    print("  3. 这就是 PyTorch 中 loss.backward() 背后在做的事情。")


if __name__ == "__main__":
    main()
