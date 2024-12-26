import numpy as np


class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

        self.W_hidden = np.random.randn(input_size, hidden_size)
        self.b_hidden = np.zeros((1, hidden_size))
        self.W_output = np.random.randn(hidden_size, output_size)
        self.b_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def forward_back(self, X, y):
        hidden_z = np.dot(X, self.W_hidden) + self.b_hidden
        hidden_a = self.sigmoid(hidden_z)
        output_z = np.dot(hidden_a, self.W_output) + self.b_output
        output_a = self.sigmoid(output_z)

        loss = self.mse(y, output_a)

        output_error = output_a - y
        output_delta = output_error * (1 - output_a) * output_a

        hidden_error = np.dot(output_delta, self.W_output.T)
        hidden_delta = hidden_error * (1 - hidden_a) * hidden_a

        self.W_output -= self.lr * np.dot(hidden_a.T, output_delta)
        self.b_output -= self.lr * np.sum(output_delta, axis=0, keepdims=True)
        self.W_hidden -= self.lr * np.dot(X.T, hidden_delta)
        self.b_hidden -= self.lr * np.sum(hidden_delta, axis=0, keepdims=True)

        return loss

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            loss = self.forward_back(X, y)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

if __name__ == '__main__':
    X = np.random.rand(100, 5)  # 100个样本，每个样本有1个特征
    y = 3 * X.sum(axis=1, keepdims=True) + np.random.randn(100, 1) * 0.1  # y = 3 * sum(x) + 噪声

    # 网络结构
    input_size = 5
    hidden_size = 10  # 中间层有10个神经元
    output_size = 1

    # 创建并训练模型
    mlp = MLP(input_size, hidden_size, output_size, lr=0.1)
    mlp.train(X, y, epochs=10000)


