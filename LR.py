import numpy as np
from matplotlib import pyplot as plt

# X：n_samples * n_features
# y: n_samples * 1
# W: n_features * 1
# y = X * W +b

class LR:
    def __init__(self, lr=0.01, epochs=10):
        self.lr = lr
        self.epochs = epochs
        self.W = None
        self.b = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def MLEloss(self, y_pre, y):
        return -np.mean(y * np.log(y_pre) + (1 - y) * np.log(1 - y_pre))

    def forward(self, X):
        output = np.dot(X, self.W) + self.b
        output = self.sigmoid(output)
        return output

    def predict(self, X):
        output = self.forward(X)
        labels_pre = [1 if i > 0.5 else 0 for i in output]
        return np.array(labels_pre)

    def trainBGD(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros(shape=(n_features, 1))
        self.b = 0

        for i in range(self.epochs):
            y_pre = self.forward(X)

            loss = self.MLEloss(y_pre, y)
            print(i, ":   loss =", loss)

            dW = (1 / n_samples) * np.dot(X.T, (y_pre - y))
            db = (1 / n_samples) * np.sum(y_pre - y)

            self.W -= self.lr * dW
            self.b -= self.lr * db


if __name__ == "__main__":
    # Generate random data
    np.random.seed(0)

    # 生成符合正态分布的数据
    np.random.seed(0)
    mean1 = np.array([2, 2])
    cov1 = np.array([[1, 0.5], [0.5, 1]])
    X1 = np.random.multivariate_normal(mean1, cov1, 100)
    y1 = np.ones((100, 1))

    mean2 = np.array([-2, -2])
    cov2 = np.array([[1, -0.5], [-0.5, 1]])
    X2 = np.random.multivariate_normal(mean2, cov2, 100)
    y2 = np.zeros((100, 1))

    # 合并数据集
    X = np.vstack((X1, X2))
    y = np.vstack((y1, y2))

    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='coolwarm', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Generated Data')
    plt.colorbar(label='Class')
    plt.show()

    # Instantiate and train the model
    model = LR(epochs=100)
    model.trainBGD(X, y)

    # Print the learned parameters
    print("W:", model.W)
    print("b:", model.b)

    # Make predictions
    new_X = np.array([2, 2])
    predictions = model.forward(new_X)
    print("Predictions:", predictions)