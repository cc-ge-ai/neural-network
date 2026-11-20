"""
Demo 02：使用 PyTorch 实现五分类神经网络

对应详细文档：docs/detailed-guide.md 第七节 Demo 02
学习目标：
    1. 用 nn.Module 定义一个神经网络。
    2. 理解 nn.Linear(5, 5) 的输入输出维度与参数量。
    3. 使用 CrossEntropyLoss、反向传播与 Adam 优化器。
    4. 划分训练集与测试集，并用准确率评估模型。

运行：
    python 04-demos/demo02-pytorch-classification.py
依赖：
    torch、numpy（未安装时脚本会给出安装命令后退出）

任务：
    输入一个五维向量，预测最大值所在的位置。
    例如 [0.1, 0.8, 0.3, 0.4, 0.2] 的最大值是 0.8，下标为 1，展示给用户时应输出 2。
"""

try:
    import numpy as np
    import torch
    import torch.nn as nn
except ModuleNotFoundError as err:
    raise SystemExit(
        f"本 Demo 需要 PyTorch，当前缺少模块：{err.name}\n"
        "安装命令（CPU 版）：pip install torch --index-url "
        "https://download.pytorch.org/whl/cpu\n"
        "如果只想先跑通原理，可以先看 03-neural-network/04-classification.py，那个版本不依赖 PyTorch。"
    )


# 定义神经网络
class TorchModel(nn.Module):

    def __init__(self):
        super(TorchModel, self).__init__()

        # 5 个输入特征，5 个输出类别
        self.linear = nn.Linear(5, 5)

    def forward(self, x):
        return self.linear(x)


# 随机生成五维向量，并给出标签：最大值所在下标
def random_simulation():
    x = np.random.random(5)

    # 获取最大值所在下标，范围 0～4
    y = x.argmax()

    return x, y


# 构建数据集
def build_dataset(total_sample_num):
    total_x = []
    total_y = []

    for _ in range(total_sample_num):
        x, y = random_simulation()
        total_x.append(x)
        total_y.append(y)

    return (
        torch.tensor(total_x, dtype=torch.float32),
        torch.tensor(total_y, dtype=torch.long),
    )


# 评估模型
def evaluate(model, test_x, test_y):
    model.eval()

    with torch.no_grad():
        y_pred = model(test_x)
        predicted = torch.argmax(y_pred, dim=1)
        accuracy = (predicted == test_y).float().mean().item()

    return accuracy


def main():

    # 固定随机种子，让每次运行结果可比
    torch.manual_seed(0)
    np.random.seed(0)

    # 训练参数
    epoch_num = 50
    batch_size = 32
    train_sample = 5000
    test_sample = 1000
    learning_rate = 0.001

    # 创建模型
    model = TorchModel()

    param_num = sum(p.numel() for p in model.parameters())
    print(f"模型参数量：{param_num}（权重 5 * 5 = 25，偏置 5）")

    # 创建优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 构建训练集和测试集
    train_x, train_y = build_dataset(train_sample)
    test_x, test_y = build_dataset(test_sample)

    print(f"训练集 {train_x.shape}，测试集 {test_x.shape}")
    print("标签范围 0～4，表示最大值下标。\n")

    # 开始训练
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0

        # 每个 epoch 遍历全部训练数据
        for start in range(0, train_sample, batch_size):
            end = min(start + batch_size, train_sample)

            x = train_x[start:end]
            y = train_y[start:end]

            # 梯度清零
            optim.zero_grad()

            # 前向传播
            y_pred = model(x)

            # 计算损失
            loss = nn.functional.cross_entropy(y_pred, y)

            # 反向传播
            loss.backward()

            # 更新参数
            optim.step()

            total_loss += loss.item() * len(y)

        avg_loss = total_loss / train_sample

        # 评估模型
        accuracy = evaluate(model, test_x, test_y)

        print(
            f"Epoch: {epoch + 1}, "
            f"Loss: {avg_loss:.4f}, "
            f"Accuracy: {accuracy:.4f}"
        )

    # 单个样本预测
    test_input = torch.tensor([[0.1, 0.8, 0.3, 0.4, 0.2]], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        output = model(test_input)
        predicted = torch.argmax(output, dim=1).item()

    print("\n输入向量：", test_input[0].tolist())
    print(f"真实最大值位置下标：{int(np.argmax(test_input.numpy()[0]))}")
    print("预测最大值位置：", predicted + 1)

    print("\n模型内部始终使用 0～4 的类别标签，只有面向用户展示时才加 1。")
    print("这个任务本身用 argmax 一行就能算完，用神经网络是为了练习训练流程。")


if __name__ == "__main__":
    main()
