"""
模块说明：反向传播（Backpropagation）—— 手算每个参数的梯度

对应详细文档：docs/detailed-guide.md 第 3.5、5.3 节第三步
学习目标：
    1. 不用任何框架，把一个两层网络的梯度完整推导并实现出来。
    2. 用数值差商校验手算梯度，确认实现正确。
    3. 看清"梯度从输出层逐段往输入层传"的过程。
    4. 理解 loss.backward() 之后参数 .grad 到底是什么。

运行：
    python 03-neural-network/03-backpropagation.py

观察要点：
    - 解析梯度与数值梯度的相对误差应在 1e-6 量级。
    - 使用自算梯度做梯度下降，Loss 会稳定下降。
    - ReLU 为 0 的位置梯度也被截断，这是网络稀疏性的来源。
"""

import numpy as np


def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    """ReLU 的导数：z > 0 时为 1，否则为 0。"""
    return (z > 0).astype(float)


def mse(y_pred, target):
    return np.mean((y_pred - target) ** 2)


def forward(X, target, W1, b1, W2, b2):
    """前向传播，同时返回反向传播需要的中间结果。"""
    z1 = X @ W1.T + b1
    a1 = relu(z1)
    z2 = a1 @ W2.T + b2
    loss = mse(z2, target)

    cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "target": target}
    return loss, cache


def backward(cache, W1, W2):
    """反向传播：从输出层往输入层逐段应用链式法则。"""
    n_samples, n_out = cache["z2"].shape

    # 第 1 段：Loss 对输出层预激活值 z2 的导数（MSE 的局部导数）
    dz2 = 2 * (cache["z2"] - cache["target"]) / (n_samples * n_out)

    # 第 2 段：z2 = a1 @ W2.T + b2
    dW2 = dz2.T @ cache["a1"]
    db2 = dz2.sum(axis=0)
    da1 = dz2 @ W2

    # 第 3 段：a1 = ReLU(z1)
    dz1 = da1 * relu_grad(cache["z1"])

    # 第 4 段：z1 = X @ W1.T + b1
    dW1 = dz1.T @ cache["X"]
    db1 = dz1.sum(axis=0)

    # 再往前就是 dX = dz1 @ W1，输入不是参数，不需要更新，但深层网络会继续往上传
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def numerical_grads(X, target, W1, b1, W2, b2, eps=1e-5):
    """逐个参数做中心差商，不依赖任何推导，用作对照基准。"""
    grads = {}
    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    for name, value in params.items():
        grad = np.zeros_like(value, dtype=float)
        it = np.nditer(value, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            origin = value[idx]

            value[idx] = origin + eps
            loss_plus = forward(X, target, W1, b1, W2, b2)[0]

            value[idx] = origin - eps
            loss_minus = forward(X, target, W1, b1, W2, b2)[0]

            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            value[idx] = origin

        grads[name] = grad

    return grads


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    np.set_printoptions(precision=5, suppress=True)

    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(4, 3))
    target = rng.normal(0, 1, size=(4, 2))

    W1 = rng.normal(0, 0.5, size=(4, 3))
    b1 = np.zeros(4)
    W2 = rng.normal(0, 0.5, size=(2, 4))
    b2 = np.zeros(2)

    section("一、要推导的四个梯度")

    print("网络：z1 = X W1^T + b1，a1 = ReLU(z1)，z2 = a1 W2^T + b2")
    print("损失：Loss = (1 / (n * m)) * sum((z2 - target)^2)")
    print("\n链式法则逐段：")
    print("  dL/dz2 = 2 (z2 - target) / (n * m)")
    print("  dL/dW2 = (dL/dz2)^T a1")
    print("  dL/db2 = sum(dL/dz2, 沿样本方向)")
    print("  dL/da1 = dL/dz2 W2")
    print("  dL/dz1 = dL/da1 * ReLU'(z1)")
    print("  dL/dW1 = (dL/dz1)^T X")
    print("  dL/db1 = sum(dL/dz1, 沿样本方向)")

    section("二、前向传播")

    loss, cache = forward(X, target, W1, b1, W2, b2)
    print(f"Loss = {loss:.6f}")
    print(f"z1 =\n{cache['z1']}")
    print(f"a1 = ReLU(z1) =\n{cache['a1']}")
    print(f"z2 =\n{cache['z2']}")

    section("三、反向传播得到的梯度")

    grads = backward(cache, W1, W2)
    for name in ["W1", "b1", "W2", "b2"]:
        print(f"dL/d{name} =\n{grads[name]}")

    print("\n注意梯度形状始终与参数形状一致：梯度告诉每个参数该往哪走、走多远。")

    section("四、用数值差商校验")

    num = numerical_grads(X, target, W1, b1, W2, b2)

    print(f"{'参数':>6}{'元素个数':>10}{'最大绝对误差':>16}{'最大相对误差':>16}")
    for name in ["W1", "b1", "W2", "b2"]:
        analytical, numerical = grads[name], num[name]
        abs_err = np.max(np.abs(analytical - numerical))
        denom = np.maximum(np.abs(numerical), 1e-8)
        rel_err = np.max(abs_err / denom)
        print(f"{name:>6}{analytical.size:>10}{abs_err:>16.3e}{rel_err:>16.3e}")

    print("\n相对误差在 1e-6 以下，说明手算公式与实现都正确。")
    print("框架的自动求导做的正是这套链式法则，只是把这些步骤自动组织了起来。")

    section("五、用自算梯度做梯度下降")

    learning_rate = 0.3
    epoch_num = 400

    print(f"{'step':>6}{'Loss':>14}")
    print(f"{0:>6}{loss:>14.6f}")

    for step in range(1, epoch_num + 1):
        loss, cache = forward(X, target, W1, b1, W2, b2)
        grads = backward(cache, W1, W2)

        W1 -= learning_rate * grads["W1"]
        b1 -= learning_rate * grads["b1"]
        W2 -= learning_rate * grads["W2"]
        b2 -= learning_rate * grads["b2"]

        if step % 40 == 0:
            print(f"{step:>6}{loss:>14.6f}")

    print("\nLoss 一路下降到接近 0，说明这 4 个样本已经被完全拟合。")
    print("梯度方向正确，整个手写训练闭环就此打通；"
          "样本太少导致 Loss 降到 0 也属于正常现象。")

    section("六、与 PyTorch 的对应关系")

    print("手写代码  ->  PyTorch 等价写法")
    for local, torch_code in [
        ("loss, cache = forward(...)", "y_pred = model(x)"),
        ("grads = backward(cache)", "loss.backward()"),
        ("W1 -= lr * grads['W1']", "optim.step()"),
        ("清空上一批的 grads 再反向传播", "optim.zero_grad()"),
    ]:
        print(f"  {local}  ->  {torch_code}")

    print("\nPyTorch 的优化器内部执行的就是 W -= lr * grad，只是加了动量等改进。")
    print("完整框架写法见 04-demos/demo02-pytorch-classification.py。")


if __name__ == "__main__":
    main()
