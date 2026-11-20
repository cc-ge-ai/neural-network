"""
模块说明：分类任务（Classification）—— Softmax 与交叉熵的完整实现

对应详细文档：docs/detailed-guide.md 第 3.7、3.8 节
学习目标：
    1. 手写数值稳定的 Softmax，把任意分数变成和为 1 的概率。
    2. 手写交叉熵 Loss，理解为什么 -log(P_true) 能惩罚错误自信。
    3. 推导 Softmax + 交叉熵的梯度，它比 MSE 简洁得多。
    4. 训练一个三分类网络，并用准确率评估。

运行：
    python 03-neural-network/04-classification.py

观察要点：
    - 参数随机初始化时往往给某个类别很高的概率，Loss 反而高于 ln(3) ~= 1.099。
    - 训练后真实类别概率接近 1，Loss 明显下降，测试准确率接近 1。
    - 梯度形式为 (预测概率 - 真实 one-hot) / 样本数，非常干净。
"""

import numpy as np

NUM_CLASSES = 3


def relu(z):
    return np.maximum(0, z)


def softmax(logits):
    """按最后一维做 Softmax；先减去最大值，避免 exp 溢出。"""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def one_hot(labels, num_classes=NUM_CLASSES):
    """把整数标签转成概率分布形式的目标。"""
    matrix = np.zeros((labels.size, num_classes))
    matrix[np.arange(labels.size), labels] = 1.0
    return matrix


def cross_entropy(probs, labels):
    """Loss = -(1/n) * sum(log(P_true))，取真实类别对应概率的负对数均值。"""
    n = labels.size
    true_prob = probs[np.arange(n), labels]
    return -np.mean(np.log(true_prob + 1e-12))


def forward(X, W1, b1, W2, b2):
    z1 = X @ W1.T + b1
    a1 = relu(z1)
    logits = a1 @ W2.T + b2
    probs = softmax(logits)
    cache = {"X": X, "z1": z1, "a1": a1, "probs": probs}
    return probs, cache


def backward(cache, labels, W2):
    n_samples = cache["X"].shape[0]

    # Softmax 与交叉熵组合后的梯度：预测概率减去真实分布
    d_logits = (cache["probs"] - one_hot(labels)) / n_samples

    dW2 = d_logits.T @ cache["a1"]
    db2 = d_logits.sum(axis=0)

    d_a1 = d_logits @ W2
    d_z1 = d_a1 * (cache["z1"] > 0)

    dW1 = d_z1.T @ cache["X"]
    db1 = d_z1.sum(axis=0)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def accuracy(probs, labels):
    return np.mean(np.argmax(probs, axis=1) == labels)


def make_data(samples_per_class=120, seed=0):
    """三个二维高斯簇，分别代表三个类别。"""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [4.0, 4.0], [0.0, 5.0]])

    X, y = [], []
    for label, center in enumerate(centers):
        X.append(rng.normal(center, 0.7, size=(samples_per_class, 2)))
        y.append(np.full(samples_per_class, label))

    X = np.vstack(X)
    y = np.concatenate(y)

    # 打乱后前 70% 训练，后 30% 测试
    index = rng.permutation(X.shape[0])
    split = int(X.shape[0] * 0.7)
    return X[index[:split]], y[index[:split]], X[index[split:]], y[index[split:]]


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    np.set_printoptions(precision=4, suppress=True)

    section("一、Softmax 把分数变成概率")

    logits = np.array([[2.0, 1.0, 0.0], [0.5, 0.5, 3.0]])
    print(f"原始分数 =\n{logits}")
    print(f"Softmax 后 =\n{softmax(logits)}")
    print(f"每行概率之和 = {softmax(logits).sum(axis=1)}")

    print("\n放大分数差距，概率会急剧集中（这是模型自信程度的来源）：")
    for scale in [0.5, 1.0, 5.0]:
        print(f"  logits * {scale}: {np.round(softmax(logits[0] * scale), 4)}")

    section("二、交叉熵如何惩罚预测")

    print(f"{'真实类别的概率':>16}{'Loss = -log(P)':>18}")
    for prob in [0.9, 0.5, 0.2, 0.1]:
        print(f"{prob:>16.2f}{-np.log(prob):>18.4f}")

    print("\n真实类别概率越高 Loss 越低；概率接近 0 时 -log 急剧上升，")
    print("所以模型'非常自信地答错'会被重罚。")
    print(f"三类完全瞎猜时 Loss 约为 ln(3) = {np.log(3):.4f}。")

    section("三、准备数据与网络")

    train_x, train_y, test_x, test_y = make_data()
    print(f"训练集 {train_x.shape[0]} 条，测试集 {test_x.shape[0]} 条，输入维度 2，类别数 3")

    rng = np.random.default_rng(1)
    W1 = rng.normal(0, 0.5, size=(8, 2))
    b1 = np.zeros(8)
    W2 = rng.normal(0, 0.5, size=(NUM_CLASSES, 8))
    b2 = np.zeros(NUM_CLASSES)
    print(f"结构：Linear(2, 8) + ReLU + Linear(8, 3)，参数量 {W1.size + b1.size + W2.size + b2.size}")

    probs, _ = forward(train_x, W1, b1, W2, b2)
    print(f"训练前：Loss = {cross_entropy(probs, train_y):.4f}，"
          f"准确率 = {accuracy(probs, train_y):.4f}")
    print("随机权重会自信地押注某个类别，因此初始 Loss 高于瞎猜水平 ln(3)。")

    section("四、训练")

    learning_rate = 0.1
    epoch_num = 1500
    print(f"\n{'epoch':>7}{'Loss':>12}{'准确率':>10}{'测试 Loss':>12}{'测试准确率':>12}")
    for epoch in range(epoch_num):
        probs, cache = forward(train_x, W1, b1, W2, b2)
        loss = cross_entropy(probs, train_y)

        grads = backward(cache, train_y, W2)
        W1 -= learning_rate * grads["W1"]
        b1 -= learning_rate * grads["b1"]
        W2 -= learning_rate * grads["W2"]
        b2 -= learning_rate * grads["b2"]

        if epoch % 150 == 0 or epoch == epoch_num - 1:
            test_probs, _ = forward(test_x, W1, b1, W2, b2)
            print(
                f"{epoch + 1:>7}{loss:>12.4f}{accuracy(probs, train_y):>10.4f}"
                f"{cross_entropy(test_probs, test_y):>12.4f}"
                f"{accuracy(test_probs, test_y):>12.4f}"
            )

    print("\n学习率取 0.1 时训练平稳；把它调到 1.0 会看到 Loss 上下震荡，反而学不好。")

    section("五、看一个样本的预测过程")

    sample = np.array([[0.3, 4.8]])
    probs, _ = forward(sample, W1, b1, W2, b2)
    print(f"输入 {sample[0]}")
    print(f"三个类别的概率 = {probs[0]}，概率之和 = {probs.sum():.4f}")
    print(f"预测类别 = {np.argmax(probs[0])}")

    print("\n三个簇的中心分别是 (0,0)、(4,4)、(0,5)，该点最靠近簇 2，预测结果与之一致。")

    section("六、Softmax + 交叉熵的梯度为什么简洁")

    print("MSE 配 Sigmoid 时梯度里会乘上 Sigmoid 的导数，两端接近 0 会导致学得很慢。")
    print("Softmax 配交叉熵时两者相乘恰好约掉，只剩：")
    print("    dL/dlogits = (预测概率 - 真实 one-hot) / 样本数")
    print("误差越大梯度越大，这与直觉完全一致，因此分类任务默认使用交叉熵。")

    print("\n下一步：04-demos/demo02-pytorch-classification.py 用 PyTorch 做同样的事。")


if __name__ == "__main__":
    main()
