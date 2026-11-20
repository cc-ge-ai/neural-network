"""
模块说明：函数（Function）—— 模型就是一个带参数的映射

对应详细文档：docs/detailed-guide.md 第 3.1 节
学习目标：
    1. 把函数理解为"输入经过规则得到输出"的一一映射。
    2. 理解模型 y_pred = f(x) 本质也是函数，参数决定映射关系。
    3. 建立"训练就是不断调整参数，让输出接近真实值"的核心直觉。

运行：
    python 01-mathematics/01-function.py

观察要点：
    - 规则不变时，输入决定输出。
    - 输入不变时，参数不同则输出不同，与真实值的差距也不同。
    - 参数从很差变到很好，Loss 就一路下降，这正是后面所有训练的基础。
"""

import numpy as np


def line(x, w=2.0, b=1.0):
    """y = w * x + b：规则固定，w 和 b 是可以替换的参数。"""
    return w * x + b


def section(title):
    print("\n" + "=" * 58)
    print(title)
    print("=" * 58)


def main():
    section("一、函数是映射：输入 -> 规则 -> 输出")

    # 最简单的函数 y = 2x + 1
    for x in [0, 1, 2, 3, 4]:
        print(f"x = {x}  ->  y = 2 * {x} + 1 = {2 * x + 1}")

    print("\n只看 x = 3 这一步：y = 2 * 3 + 1 = 7。")
    print("输入确定以后输出唯一确定，这就是映射的含义。")
    print("把一整段输入都算出来连成线，图像就是一条直线。")

    section("二、模型也是一种函数：y_pred = f(x)")

    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    # 假设真实规律是 y = 2x + 1，把它当成"数据告诉我们的答案"
    y_true = line(x, w=2.0, b=1.0)

    print("把真实规律写成函数形式：")
    print(f"{'x':>6}{'y_true':>10}")
    for xi, yi in zip(x, y_true):
        print(f"{xi:>6.1f}{yi:>10.1f}")

    print("\n机器学习里通常写作：")
    print("    y_pred = f(x)")
    print("其中 x 是输入数据，f 是模型，y_pred 是模型给出的预测结果。")

    section("三、参数不同，同一个模型就是不同的函数")

    print(f"{'参数':>16}{'预测值':>26}{'平均差距':>12}")
    for w, b in [(0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (1.5, 3.0)]:
        y_pred = line(x, w=w, b=b)

        # 平均绝对差距，是最朴素的一种 Loss
        gap = np.mean(np.abs(y_pred - y_true))

        print(
            f"{'w=%s b=%s' % (w, b):>16}"
            f"{str(np.round(y_pred, 2)):>26}"
            f"{gap:>12.3f}"
        )

    print("\n结论：")
    print("  1. 固定输入 x，只改动 w 和 b，输出就完全不同。")
    print("  2. w = 2, b = 1 时差距为 0，说明这组参数正好还原了数据规律。")
    print("  3. 所谓训练模型，就是自动地搜索出这样一组参数。")

    section("四、模型可以有多复杂，但形式始终是 f(x)")

    def MLP(x, w1, b1, w2, b2):
        """两层模型：先做一次线性变换，再做一次线性变换。"""
        h = w1 * x + b1
        return w2 * h + b2

    y_pred = MLP(np.array([1.0, 2.0]), w1=3.0, b1=1.0, w2=2.0, b2=5.0)
    print(f"两层模型 f(x) = 2 * (3x + 1) + 5，输入 [1, 2] -> 输出 {y_pred}")
    print("层数再多、参数再多，也仍然是'输入经过参数化的规则得到输出'。")


if __name__ == "__main__":
    main()
