import numpy as np


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return out


def CE_loss(y_true, y_pred):
    out = -np.sum(y_true * np.log(y_pred), axis=-1)
    return out


if __name__ == '__main__':
    # 定义输入向量
    logits = np.array([[2.0, 1.0, 0.1],
                       [1.0, 3.0, 0.2]])
    y_true = np.array([[1, 0, 0],
                       [0, 1, 0]])

    # 计算 softmax 概率分布
    probabilities = softmax(logits)
    print("Softmax probabilities:\n", probabilities)

    # 计算交叉熵损失
    loss = CE_loss(y_true, probabilities)
    print("Cross-entropy loss:\n", loss)