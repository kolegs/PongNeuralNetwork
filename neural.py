import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, size):
        self.size = size
        self.w = []
        self.b = []
        self.a = []
        self.x = []
        self.delta = []
        self.cost = []
        self.learning_num = 0
        self.learning_rate = 0.001

        for i in range(1, len(size)):
            # Xavier initialization, which is recommended for sigmoid function
            # https://hackernoon.com/how-to-initialize-weights-in-a-neural-net-so-it-performs-well-3e9302d4490f
            # self.w.append(np.random.randn(size[i], size[i - 1]) * np.sqrt(1/(size[i] + size[i - 1])))

            # Random initialization
            self.w.append(np.random.randn(size[i], size[i - 1]))
            self.b.append(np.zeros((size[i], 1)))

            self.a.append(np.zeros((size[i], 1)))
            self.x.append(np.zeros((size[i], 1)))
            self.delta.append(np.zeros((size[i], 1)))

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def cost(self, X, Y):
        f = self.forward_propagation(X)
        return np.sum(np.power((f - Y), 2)) / 2

    def forward_propagation(self, X):
        a = X
        for i in range(1, len(self.size)):
            self.x[i - 1] = a
            a = self.sigmoid(np.dot(self.w[i - 1], a) + self.b[i - 1])
            self.a[i - 1] = a
        return a

    def backward_propagation(self, Y, A):
        for i in reversed(range(1, len(self.size))):
            if i == len(self.size) - 1:
                self.delta[i - 1] = (A - Y) * (A * (1 - A)).reshape(self.size[-1], 1)
            else:
                self.delta[i - 1] = np.dot(self.w[i].T, self.delta[i]) * (self.a[i - 1] * (1 - self.a[i - 1]))

    def learn(self):
        for i in range(1, len(self.size)):
            self.w[i - 1] = self.w[i - 1] - self.learning_rate * self.delta[i - 1] * self.x[i - 1].T
            self.b[i - 1] = self.b[i - 1] - self.learning_rate * self.delta[i - 1]

    def learning_rate_decay(self):
        self.learning_rate = self.learning_rate * 0.95
        # if self.learning_rate < 0.001:
        #     self.learning_rate = 0.001

    def feed(self, X, Y):
        # for _ in range(20):
        for i in range(Y.shape[1]):
            self.learning_num = self.learning_num + 1
            if i % 10000 == 0:
                print(f"Progress learning {i} / {Y.shape[1]}")
            if self.learning_num % 100000 == 0 or self.learning_num == 1:
                current_cost = self.compute_cost(X, Y)
                print(f"Cost after {self.learning_num}: {current_cost}")
                self.cost.append(current_cost)
            if self.learning_num % 500000 == 0:
                self.learning_rate_decay()
            j = np.random.randint(int(Y.shape[1]))
            currentX = X[:, j].reshape(4, 1)
            currentY = Y[:, j].reshape(1, 1)
            A = self.forward_propagation(currentX)
            self.backward_propagation(currentY, A)
            self.learn()

    def compute_cost(self, X, Y):
        Y_hat = self.forward_propagation(X)
        diff = Y_hat - Y
        cost = (1 / 2) * np.sum(diff * diff)
        return cost

    def show_cost(self):
        plt.plot(self.cost)
        plt.ylabel('Cost')
        plt.savefig("plot.png")
        plt.show()


