import numpy as np


# X：n_samples * n_features
# y: n_samples * 1
# W: n_features * 1
# y = X * W +b

class LinearReg:
    def __init__(self, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.W = None
        self.b = None

    def mse(self, y_pre, y):
        return np.mean((y_pre - y) ** 2)

    def forward(self, X):
        return np.dot(X, self.W) + self.b

    def trainBGD(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros(shape=(n_features, 1))
        self.b = 0

        for i in range(self.epochs):
            y_pre = self.forward(X)

            loss = self.mse(y_pre, y)
            print(i, ":   loss =", loss)

            dW = (1 / n_samples) * np.dot(X.T, (y_pre - y))
            db = (1 / n_samples) * np.sum(y_pre - y)
            
            self.W -= self.lr * dW
            self.b -= self.lr * db


if __name__ == "__main__":
    # Generate random data
    np.random.seed(0)
    X = np.array([[2, 1], [4, 4], [6, 5]])
    y = np.array([1, 2, 3]).reshape(-1, 1)

    # Instantiate and train the model
    model = LinearReg()
    model.trainBGD(X, y)

    # Print the learned parameters
    print("W:", model.W)
    print("b:", model.b)

    # Make predictions
    new_X = np.array([6, 5])
    predictions = model.forward(new_X)
    print("Predictions:", predictions)
