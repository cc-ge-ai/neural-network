"""
模块说明：线性回归（Linear Regression）—— 最小完整的机器学习流程

对应详细文档：docs/detailed-guide.md 第 4.1 至 4.3 节
学习目标：
    1. 对比传统编程与机器学习：规则从哪里来。
    2. 亲手划分训练集与测试集，理解模型评估。
    3. 走通"模型 -> Loss -> 梯度 -> 更新"这一最小闭环。
    4. 用 MSE 和 R^2 判断模型是否真的学到了规律。

运行：
    python 02-machine-learning/01-linear-regression.py

观察要点：
    - 训练只使用训练集，测试集只在最后评估一次。
    - 训练 Loss 持续下降，同时测试 Loss 也下降，说明没有过拟合。
    - 最终 w 接近 2、b 接近 3，即数据背后的真实规律。
"""

import numpy as np


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def mean_squared_error(y_pred, y_true):
    """回归任务的损失函数：Loss = (1/n) * sum((y_pred - y_true)^2)。"""
    return np.mean((y_pred - y_true) ** 2)


def r_squared(y_pred, y_true):
    """R^2 衡量模型比'直接猜平均值'好多少，越接近 1 越好。"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def main():
    section("一、传统编程与机器学习的区别")

    print("传统编程：程序员写出规则。")
    print("    输入面积 -> 规则(面积 * 单价 + 楼层修正) -> 输出房价")
    print("\n机器学习：给出大量(面积, 房价)数据，规则由参数自己拟合出来。")
    print("    输入面积 + 正确房价 -> 调整 w 和 b -> 得到能预测新数据的模型")

    print("\n本模块的数据由下面这条规则生成，但训练时假装不知道它：")
    print("    y = 2 * x + 3 + 噪声")

    section("二、准备数据并划分训练集与测试集")

    np.random.seed(42)

    x_all = np.linspace(0, 10, 100)
    y_all = 2 * x_all + 3 + np.random.normal(0, 1, x_all.size)

    # shuffle 后前 80 个作训练集，后 20 个作测试集
    index = np.random.permutation(x_all.size)
    train_index, test_index = index[:80], index[80:]

    train_x, train_y = x_all[train_index], y_all[train_index]
    test_x, test_y = x_all[test_index], y_all[test_index]

    print(f"训练集 {train_x.size} 条，用于调整参数")
    print(f"测试集 {test_x.size} 条，模型训练时完全看不到，只用于最后评估")
    print("划分的目的：检验模型学到的是规律，还是背下了答案。")

    section("三、模型与初始参数")

    w, b = 0.0, 0.0
    print(f"模型：y_pred = w * x + b")
    print(f"初始参数：w = {w}, b = {b}（随机或全零初始化，此时预测几乎无意义）")

    section("四、训练：重复执行前向传播、计算 Loss、求梯度、更新参数")

    learning_rate = 0.01
    epoch_num = 1000

    print(f"\n{'epoch':>7}{'train Loss':>14}{'test Loss':>12}{'w':>10}{'b':>10}")
    for epoch in range(epoch_num):

        # 1. 前向传播
        y_pred = w * train_x + b

        # 2. 计算 Loss
        loss = mean_squared_error(y_pred, train_y)

        # 3. 计算梯度（推导见 docs/detailed-guide.md 第 4.3、4.4 节）
        dw = np.mean(2 * (y_pred - train_y) * train_x)
        db = np.mean(2 * (y_pred - train_y))

        # 4. 更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if epoch % 100 == 0 or epoch == epoch_num - 1:
            test_loss = mean_squared_error(w * test_x + b, test_y)
            print(f"{epoch + 1:>7}{loss:>14.4f}{test_loss:>12.4f}{w:>10.4f}{b:>10.4f}")

    print("\n训练集 Loss 与测试集 Loss 同步下降，说明模型学到的是通用规律。")

    section("五、评估模型")

    train_pred = w * train_x + b
    test_pred = w * test_x + b

    print(f"学到的参数：w = {w:.4f}, b = {b:.4f}（真实值 w = 2, b = 3）")
    print(f"测试集 MSE ：{mean_squared_error(test_pred, test_y):.4f}")
    print(f"测试集 R^2 ：{r_squared(test_pred, test_y):.4f}")
    print("R^2 接近 1 表示模型解释绝大部分变化；若只猜平均值，R^2 为 0。")

    section("六、看看模型对新数据的预测")

    print(f"{'输入 x':>10}{'真实 y':>12}{'预测 y':>12}{'误差':>10}")
    for i in test_index[:8]:
        xi, yi = x_all[i], y_all[i]
        pred = w * xi + b
        print(f"{xi:>10.2f}{yi:>12.3f}{pred:>12.3f}{pred - yi:>10.3f}")

    print("\n误差主要来自生成数据时加入的噪声，这部分是无法预测的随机成分。")

    print("\n下一步：04-demos/demo01-numpy-linear-regression.py 会把这些步骤配上绘图。")


if __name__ == "__main__":
    main()
