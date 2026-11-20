"""
Demo 01：使用 NumPy 手动实现线性回归

对应详细文档：docs/detailed-guide.md 第七节 Demo 01
学习目标：
    1. 不使用任何深度学习框架，自己实现一个最简单的机器学习模型。
    2. 亲手写出模型、Loss、梯度、参数更新这四步。
    3. 观察参数如何从 0 逐渐逼近真实规律 y = 2x + 3。

运行：
    python 04-demos/demo01-numpy-linear-regression.py
依赖：
    numpy，matplotlib 可选（缺失时跳过绘图，不影响训练结果打印）

重点公式：
    模型      y_pred = w x + b
    损失      Loss   = (1/n) sum (y_pred - y)^2
    权重梯度  dL/dw  = (2/n) sum (y_pred - y) x
    偏置梯度  dL/db  = (2/n) sum (y_pred - y)
    参数更新  w = w - lr * dL/dw，b = b - lr * dL/db
"""

from pathlib import Path

import numpy as np

# 生成训练数据：真实规律是 y = 2x + 3，再加一点噪声
np.random.seed(42)

X = np.linspace(0, 10, 100)
Y = 2 * X + 3 + np.random.normal(0, 1, 100)

# 学习率：决定每一步调整参数多大
LEARNING_RATE = 0.01

# 训练轮数
EPOCH_NUM = 1000

OUTPUT_DIR = Path(__file__).parent / "output"


def train(x, y, learning_rate=LEARNING_RATE, epoch_num=EPOCH_NUM):
    """梯度下降主循环，返回学到的参数与每轮 Loss。"""

    # 初始化模型参数：故意从 0 开始，方便观察学习过程
    w = 0.0
    b = 0.0
    loss_list = []

    for epoch in range(epoch_num):

        # 1. 前向传播
        y_pred = w * x + b

        # 2. 计算 Loss（均方误差）
        loss = np.mean((y_pred - y) ** 2)

        # 3. 计算梯度
        dw = np.mean(2 * (y_pred - y) * x)
        db = np.mean(2 * (y_pred - y))

        # 4. 更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db

        loss_list.append(loss)

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch: {epoch + 1}, "
                f"Loss: {loss:.4f}, "
                f"w: {w:.4f}, "
                f"b: {b:.4f}"
            )

    return w, b, loss_list


def plot(w, b, loss_list):
    """有显示环境时弹窗，没有时保存成文件，两种情况下都能看到结果。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n未安装 matplotlib，已跳过绘图。需要看图时执行：pip install matplotlib")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 绘制 Loss 曲线
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    loss_path = OUTPUT_DIR / "demo01-loss.png"
    plt.savefig(loss_path, bbox_inches="tight")
    plt.show()

    # 绘制拟合结果
    plt.figure()
    plt.scatter(X, Y, label="Original Data")
    plt.plot(X, w * X + b, label="Prediction")
    plt.legend()
    fit_path = OUTPUT_DIR / "demo01-fit.png"
    plt.savefig(fit_path, bbox_inches="tight")
    plt.show()

    print(f"图片已保存：{loss_path.name}、{fit_path.name}（目录 {OUTPUT_DIR}）")


def main():
    w, b, loss_list = train(X, Y)

    print("最终权重 w：", w)
    print("最终偏置 b：", b)

    print(f"\n真实规律 y = 2x + 3，模型学到 w = {w:.4f}、b = {b:.4f}")
    print(f"参数误差：|dw| = {abs(w - 2):.4f}，|db| = {abs(b - 3):.4f}")
    print(f"Loss 从 {loss_list[0]:.4f} 降到 {loss_list[-1]:.4f}")

    print("\n初始参数为 0，模型并不知道答案，只靠不断降低 Loss 自己逼近了这条直线。")

    plot(w, b, loss_list)


if __name__ == "__main__":
    main()
