import numpy as np


def auc(y, pred):
    true = 0  # 分子
    all = 0  # 分母
    n = len(y)

    for i in range(n - 1):
        for j in range(i + 1, n):
            if y[i] != y[j]:
                all += 1
                # 统计所有正负样本对中，模型把相对位置排序正确的数量
                if (y[i] > y[j] and pred[i] > pred[j]) or (y[i] < y[j] and pred[i] < pred[j]):
                    true += 1
    return true / all


if __name__ == '__main__':
    ##给定的真实y 和 预测pred
    y = np.array([1, 0, 0, 0, 1, 0, 1, 0, 0, 1])
    pred = np.array([0.9, 0.4, 0.3, 0.1, 0.35, 0.6, 0.65, 0.32, 0.8, 0.7])
    res = auc(y, pred)
    print("AUC =", res)
