import numpy as np


def _activate(x):
    return 2. / (1 + np.exp(x)) - 1


def _derivative_activate(phi_x):
    return (1 + phi_x) * (1 - phi_x) / 2


class SingleLayerPerceptron:
    """Single-layer Perceptron (neural network with one layer of McCulloch Pitts neurons)."""

    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

    def learn(self, patterns, targets):
        patterns = np.concatenate(patterns, np.ones(patterns.shape[1]), axis=0)     # add extra row for bias term
        self.weights = 0.01 * np.random.randn(targets.shape[0], patterns.shape[0])  # init weights
        for i in range(self.epochs):
            self.weights += self.learning_rate * (targets - self.weights @ patterns) @ patterns.T


class TwoLayerPerceptron:
    """Two-layer Perceptron (neural network with two layers of McCulloch Pitts neurons)."""

    def __init__(self, learning_rate, epochs, hidden_nodes, alpha):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_nodes = hidden_nodes    # number of neurons in the hidden layer
        self.alpha = alpha                  # factor for momentum term
        self.W = None                       # weights first layer
        self.V = None                       # weights second layer

    def learn(self, patterns, targets):
        # init
        patterns = np.concatenate(patterns, np.ones(patterns.shape[1]), axis=0)     # add extra row for bias term
        self.W = 0.01 * np.random.randn(self.hidden_nodes, patterns.shape[0])
        self.V = 0.01 * np.random.randn(targets.shape[0], self.hidden_nodes+1)      # +1 for the bias weight
        dW = 0
        dV = 0

        for i in range(self.epochs):
            # forward pass
            H = _activate(self.W @ patterns)
            H = np.concatenate(H, np.ones(patterns.shape[0]), axis=0)               # add extra row for bias term
            O = _activate(self.V @ H)

            # backward pass
            delta_o = (O - targets) * _derivative_activate(O)
            delta_h = (self.V.T @ delta_o) * _derivative_activate(H)
            delta_h = delta_h[:-1]                                                  # remove extra row

            # weight update
            dW = self.alpha * dV - (1 - self.alpha) * (delta_o @ patterns.T)
            dV = self.alpha * dW - (1-self.alpha) * (delta_h @ H.T)
            self.W += self.learning_rate * dW
            self.V += self.learning_rate * dV
