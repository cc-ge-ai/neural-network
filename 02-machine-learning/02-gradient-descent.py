"""
模块说明：梯度下降（Gradient Descent）—— 学习率如何决定成败

对应详细文档：docs/detailed-guide.md 第 4.4 节
学习目标：
    1. 用一个一维函数观察参数沿负梯度方向逐步靠近最低点。
    2. 直观看到学习率过小收敛慢、过大来回震荡甚至发散。
    3. 理解批量梯度下降、随机梯度下降与小批量梯度下降的差别。
    4. 明白 PyTorch 里 optim.step() 做的事就是这一行更新公式。

运行：
    python 02-machine-learning/02-gradient-descent.py

观察要点：
    - 更新公式：w_new = w_old - lr * dLoss/dw。
    - 学习率 0.5 时一步落到最低点；1.1 时 Loss 迅速爆炸。
    - 随机梯度下降单步 Loss 抖动很大，但整体趋势一致。
"""

import numpy as np


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def ascii_curve(values, width=44):
    """把一列 Loss 用字符画出来，不依赖 matplotlib 也能看到趋势。"""
    values = np.asarray(values, dtype=float)
    lo, hi = values.min(), values.max()
    span = hi - lo

    for i, v in enumerate(values):
        filled = int((v - lo) / span * width) if span > 0 else width
        print(f"{i:>4} {'#' * max(filled, 1):<{width}}{v:>12.4f}")


def loss_1d(w):
    """目标函数 Loss = (w - 5)^2，最低点在 w = 5。"""
    return (w - 5) ** 2


def grad_1d(w):
    return 2 * (w - 5)


def main():
    section("一、沿着负梯度方向走")

    print("目标函数 Loss = (w - 5)^2，梯度 dLoss/dw = 2(w - 5)。")
    print("起点 w = 0，梯度为 -10，负梯度方向就是 w 增大的方向。")

    w, lr = 0.0, 0.1
    print(f"\n{'step':>6}{'w':>10}{'梯度':>12}{'Loss':>12}")
    for step in range(11):
        print(f"{step:>6}{w:>10.4f}{grad_1d(w):>12.4f}{loss_1d(w):>12.4f}")
        w = w - lr * grad_1d(w)

    print("\n梯度绝对值越来越小，Loss 越来越低，最后停在 w = 5 附近。")

    section("二、学习率对比")

    print(f"{'学习率':>8}{'20 步后 w':>14}{'20 步后 Loss':>16}{'结果':>14}")
    for lr in [0.001, 0.05, 0.1, 0.5, 1.1]:
        w = 0.0
        for _ in range(20):
            w = w - lr * grad_1d(w)

        loss = loss_1d(w)
        if not np.isfinite(loss):
            result = "溢出"
        elif loss < 1e-6:
            result = "收敛"
        elif loss > 100:
            result = "发散"
        else:
            result = "仍在下降"
        print(f"{lr:>8}{w:>14.4f}{loss:>16.4f}{result:>14}")

    print("\n结论：")
    print("  0.001 每步只移动 0.02，20 步远远不够；")
    print("  0.5 恰好抵消，一步落到最低点（更新变为 w - 1.0*(w-5) = 5）；")
    print("  1.1 步长超过最低点，来回跳且幅度越来越大，直接发散。")

    section("三、把收敛过程画出来")

    for lr in [0.01, 0.1, 0.45]:
        print(f"\n学习率 {lr} 的 Loss 轨迹（前 20 步）：")
        w, trace = 0.0, []
        for _ in range(20):
            trace.append(loss_1d(w))
            w = w - lr * grad_1d(w)
        ascii_curve(trace)

    section("四、批量、随机与小批量梯度下降")

    np.random.seed(0)
    train_x = np.linspace(0, 10, 200)
    train_y = 2 * train_x + 3 + np.random.normal(0, 1, train_x.size)

    def full_batch(steps, lr):
        w = b = 0.0
        for _ in range(steps):
            y_pred = w * train_x + b
            dw = np.mean(2 * (y_pred - train_y) * train_x)
            db = np.mean(2 * (y_pred - train_y))
            w -= lr * dw
            b -= lr * db
        return w, b

    def stochastic(steps, lr):
        w = b = 0.0
        for i in range(steps):
            j = i % train_x.size
            error = w * train_x[j] + b - train_y[j]
            dw = 2 * error * train_x[j]
            db = 2 * error
            w -= lr * dw
            b -= lr * db
        return w, b

    def mini_batch(steps, lr, batch_size=32):
        w = b = 0.0
        for i in range(steps):
            start = (i * batch_size) % train_x.size
            end = min(start + batch_size, train_x.size)
            xs, ys = train_x[start:end], train_y[start:end]
            error = w * xs + b - ys
            dw = np.mean(2 * error * xs)
            db = np.mean(2 * error)
            w -= lr * dw
            b -= lr * db
        return w, b

    def mse(w, b):
        """整批数据的均方误差，用于横向比较三种优化方式。"""
        residual = w * train_x + b - train_y
        return np.mean(residual**2)

    # 按"实际使用过的样本数"而不是更新次数来比较，才公平
    print(f"{'方式':>12}{'更新次数':>10}{'用过样本数':>14}{'w':>9}{'b':>9}{'Loss':>10}")
    for name, (w, b), updates, used in [
        ("批量", full_batch(300, 0.01), 300, 300 * train_x.size),
        ("随机", stochastic(300, 0.001), 300, 300),
        ("随机", stochastic(60000, 0.001), 60000, 60000),
        ("小批量", mini_batch(300, 0.01), 300, 300 * 32),
    ]:
        print(f"{name:>12}{updates:>10}{used:>14}{w:>9.4f}{b:>9.4f}{mse(w, b):>10.4f}")

    print("\n批量：每步用全部数据，梯度稳定，但一次计算量大。")
    print("随机：每步只用一个样本，单步便宜、抖动明显，需要更小的学习率。")
    print("小批量：两者折中，是深度学习实际最常用的方式（Demo 02 用的就是它）。")
    print("同样用掉 6 万个样本，随机梯度下降已达到与批量相近的 Loss，但每步只算一条数据。")
    print("小批量只用了 9600 个样本就接近收敛，兼顾稳定与速度，所以成为默认选择。")


if __name__ == "__main__":
    main()
