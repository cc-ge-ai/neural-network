"""
模块说明：导数（Derivative）—— 描述函数在某一点的变化率

对应详细文档：docs/detailed-guide.md 第 3.2 节
学习目标：
    1. 理解导数是"输入变化一点点，输出变化多少倍"的比值。
    2. 会用差商 (f(x+h) - f(x)) / h 从数值上逼近导数。
    3. 理解导数依赖位置：同一条曲线在不同 x 处变化率不同。
    4. 建立"参数变化时 Loss 如何变化"这一机器学习中的关键视角。

运行：
    python 01-mathematics/02-derivative.py

观察要点：
    - h 越小，数值导数越接近解析导数。
    - h 过小时结果反而变差，因为浮点精度有限。
    - 导数为负表示参数增大时函数值下降。
"""

import numpy as np


def numerical_derivative(f, x, h):
    """用差商近似导数：割线斜率随着 h 变小而趋近切线斜率。"""
    return (f(x + h) - f(x)) / h


def central_difference(f, x, h=1e-5):
    """中心差商，比单侧差商更接近真实导数。"""
    return (f(x + h) - f(x - h)) / (2 * h)


def section(title):
    print("\n" + "=" * 58)
    print(title)
    print("=" * 58)


def main():
    section("一、导数就是变化率")

    # y = x^2，解析导数 dy/dx = 2x
    def square(x):
        return x**2

    print("函数 y = x^2，解析导数 dy/dx = 2x。")
    print(f"{'x':>6}{'2x':>10}")
    for x in [0, 1, 2, 3, 4]:
        print(f"{x:>6}{2 * x:>10}")

    print("\nx = 3 时导数为 6：在 x = 3 附近，输入每变化一点，输出大约变化 6 倍。")
    print("导数随 x 变化，说明曲线各处的陡峭程度不同，这一点后面理解梯度很重要。")

    section("二、用差商把导数算出来")

    x0 = 3.0
    print(f"在 x = {x0} 处逐步缩小 h，观察数值导数如何趋近 6：")
    print(f"{'h':>12}{'数值导数':>16}{'与 6 的误差':>16}")

    for h in [1e-1, 1e-2, 1e-3, 1e-5, 1e-7, 1e-10]:
        d = numerical_derivative(square, x0, h)
        print(f"{h:>12.0e}{d:>16.8f}{abs(d - 6):>16.2e}")

    print("\n注意 h = 1e-10 时误差反而变大：")
    print("  f(x+h) 与 f(x) 太接近，两个浮点数相减会丢掉大量有效位数。")
    print("  实际项目不手算导数，而是交给框架的自动求导机制。")

    section("三、导数如何解释'变化一点点'")

    dx = 0.01
    predict = 6 * dx
    real = square(x0 + dx) - square(x0)
    print("x 从 3 增加到 3.01：")
    print(f"  用导数估计的变化量 dy ~= 6 * 0.01 = {predict:.5f}")
    print(f"  真实变化量 dy = 3.01^2 - 3^2 = {real:.5f}")
    print("  两者非常接近，这就是导数作为局部线性近似的意义。")

    section("四、常数与线性函数的导数")

    def f(x):
        return 2 * x + 1

    print("函数 y = 2x + 1：")
    print(f"{'x':>6}{'数值导数':>14}")
    for x in [-1, 0, 1, 2, 3]:
        print(f"{x:>6}{central_difference(f, x):>14.5f}")
    print("导数处处等于 2，因为直线的倾斜程度在任何位置都一样。")

    section("五、导数在机器学习中的作用")

    print("把 Loss 看作参数 w 的函数，导数 dLoss/dw 回答的是：")
    print("  w 增大一点点，Loss 会变大还是变小、变化幅度有多大。")
    print("\n例如 Loss = (w - 2)^2，真实答案 w = 2：")

    def loss(w):
        return (w - 2) ** 2

    print(f"{'w':>8}{'Loss':>10}{'dLoss/dw':>12}{'含义':>22}")
    for w in [0.0, 1.0, 1.9, 2.0, 3.0]:
        d = central_difference(loss, w)
        if abs(d) < 1e-6:
            meaning = "已在最低点"
        elif d > 0:
            meaning = "w 应减小"
        else:
            meaning = "w 应增大"
        print(f"{w:>8.1f}{loss(w):>10.3f}{d:>12.4f}{meaning:>22}")

    print("\n梯度下降正是利用这一点：沿导数反方向调整参数，Loss 就会下降。")


if __name__ == "__main__":
    main()
