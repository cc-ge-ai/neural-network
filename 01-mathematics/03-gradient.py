"""
模块说明：梯度（Gradient）—— 多个偏导数组成的向量

对应详细文档：docs/detailed-guide.md 第 3.3、3.4 节
学习目标：
    1. 理解偏导数：固定其他参数，只看某一个参数对 Loss 的影响。
    2. 理解梯度是所有偏导数排成的向量，指向 Loss 增长最快的方向。
    3. 会用数值方法（中心差商）计算梯度，并与解析结果对照。
    4. 用梯度下降最小化一个二元函数，完整跑通"多参数优化"。

运行：
    python 01-mathematics/03-gradient.py

观察要点：
    - L = w1^2 + 2*w2^2 的梯度为 [2w1, 4w2]。
    - 梯度为 0 的点是最低点 (0, 0)。
    - 沿着负梯度方向走，Loss 一定下降；沿着梯度方向走，上升最快。
"""

import numpy as np


def loss(w):
    """L = w1^2 + 2 * w2^2，最低点在 (0, 0)。"""
    return w[0] ** 2 + 2 * w[1] ** 2


def analytical_gradient(w):
    """解析梯度：dL/dw1 = 2w1，dL/dw2 = 4w2。"""
    return np.array([2 * w[0], 4 * w[1]])


def numerical_gradient(f, w, eps=1e-5):
    """逐维中心差商，不推导公式也能得到梯度，是检查代码的常用手段。"""
    grad = np.zeros_like(w, dtype=float)
    for i in range(len(w)):
        plus = w.copy()
        minus = w.copy()
        plus[i] += eps
        minus[i] -= eps
        grad[i] = (f(plus) - f(minus)) / (2 * eps)
    return grad


def section(title):
    print("\n" + "=" * 58)
    print(title)
    print("=" * 58)


def main():
    section("一、偏导数：一次只看一个参数")

    w = np.array([1.0, 2.0])
    print(f"取 w1 = {w[0]}, w2 = {w[1]}，此时 L = {loss(w):.2f}")

    step = 0.01
    for i, name in [(0, "w1"), (1, "w2")]:
        changed = w.copy()
        changed[i] += step
        delta = (loss(changed) - loss(w)) / step
        print(f"  只把 {name} 增大 {step}，Loss 平均每单位变化 {delta:.3f}")

    print("\n所谓偏导数，就是只动一个参数时 Loss 的变化率。")
    print("神经网络有成千上万个参数，训练时需要对每个参数都算一个偏导数。")

    section("二、解析梯度与数值梯度对照")

    for w in [np.array([1.0, 2.0]), np.array([-3.0, 0.5]), np.array([0.0, 0.0])]:
        ga = analytical_gradient(w)
        gn = numerical_gradient(loss, w)
        print(f"w = {w}  解析梯度 {ga}  数值梯度 {np.round(gn, 5)}")

    print("\n两者一致，说明公式与实现都对；数值梯度就是自动求导的对照基准。")

    section("三、梯度指向 Loss 增长最快的方向")

    w = np.array([1.0, 2.0])
    g = analytical_gradient(w)
    g_norm = g / np.linalg.norm(g)

    print(f"w = {w}，梯度 = {g}，单位梯度方向 = {np.round(g_norm, 4)}")
    print(f"{'方向':>22}{'单位向量':>20}{'方向导数':>12}")
    for name, u in [
        ("单位梯度方向", g_norm),
        ("负梯度方向", -g_norm),
        ("w1 轴方向", np.array([1.0, 0.0])),
        ("w2 轴方向", np.array([0.0, 1.0])),
        ("垂直梯度方向", np.array([-g_norm[1], g_norm[0]])),
    ]:
        # 方向导数 = 梯度与该单位方向的内积
        print(f"{name:>22}{str(np.round(u, 4)):>20}{g @ u:>12.4f}")

    print("\n可以看到：")
    print("  沿梯度方向方向导数最大 -> 上升最快；")
    print("  沿负梯度方向方向导数最小 -> 下降最快；")
    print("  垂直于梯度的方向方向导数为 0 -> 走这一步 Loss 基本不变。")

    section("四、梯度下降：让参数自己走到最低点")

    w = np.array([3.0, 3.0])
    lr = 0.1
    print(f"起点 w = {w}，学习率 {lr}，更新公式 w = w - lr * gradient")
    print(f"{'step':>6}{'w1':>10}{'w2':>10}{'Loss':>12}{'梯度范数':>12}")

    history = []
    for step in range(31):
        g = analytical_gradient(w)
        current_loss = loss(w)
        history.append(current_loss)

        if step % 3 == 0 or step == 30:
            print(
                f"{step:>6}{w[0]:>10.5f}{w[1]:>10.5f}"
                f"{current_loss:>12.6f}{np.linalg.norm(g):>12.6f}"
            )

        w = w - lr * g

    print(f"\n最终 w = {np.round(w, 5)}，Loss = {loss(w):.3e}")
    print("梯度范数不断变小，说明越来越接近平坦的最低点 (0, 0)。")

    section("五、学习率决定步长")

    for lr in [0.01, 0.1, 0.4, 0.6]:
        w = np.array([3.0, 3.0])
        for _ in range(30):
            w = w - lr * analytical_gradient(w)
        print(f"学习率 {lr:>4}：30 步后 Loss = {loss(w):.6f}")

    print("\n学习率太小则走得慢，适中则快速收敛。")
    print("w2 方向的二阶导数是 4，学习率必须小于 2 / 4 = 0.5，0.6 已经越过上限：")
    print("参数会来回震荡并迅速发散，Loss 不降反升。")
    print("梯度下降的完整讨论见 02-machine-learning/02-gradient-descent.py。")


if __name__ == "__main__":
    main()
