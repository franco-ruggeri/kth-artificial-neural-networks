import numpy as np
from utility import plot_decision_boundary


class SingleLayerPerceptron:
    """Single-layer perceptron (neural network with one layer of McCulloch Pitts neurons)."""

    def __init__(self, learning_rate, epochs, method='perceptron', mode='batch', bias=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.bias = bias
        self.weights = None
        self.mce = None

        # select method
        if method == 'delta':
            self.delta = True
        elif method == 'perceptron':
            self.delta = False
        else:
            exit(-1)

        # select mode
        if mode == 'batch':
            self.batch = True
        elif mode == 'sequential':
            self.batch = False
        else:
            exit(-1)

    def learn(self, patterns, targets):
        """Train the perceptron using the Delta learning rule."""
        if self.bias:
            patterns = np.concatenate((patterns, np.ones((1, patterns.shape[1]))), axis=0)  # add row for bias term
        self.weights = self._sigma * np.random.randn(targets.shape[0], patterns.shape[0])   # init weights

        # stats
        y = self.predict(patterns)
        self.mce = [len(np.where(targets != y)[0]) / patterns.shape[1]]                     # misclassification error

        for i in range(self.epochs):
            # animation of decision boundary
            # xlim = [min(patterns[0]) - 0.1, max(patterns[0]) + 0.1, min(patterns[1]) - 0.1, max(patterns[1]) + 0.1]
            # plot_decision_boundary(self, xlim, 'y')

            # update weights
            if self.batch:
                if self.delta:
                    e = targets - self.weights @ patterns
                else:
                    e = targets - self.predict(patterns)
                self.weights += self.learning_rate * e @ patterns.T
            else:
                for n in range(patterns.shape[1]):
                    if self.delta:
                        e = targets[:, n] - self.weights @ patterns[:, n]
                    else:
                        e = targets[:, n] - self.predict(patterns[:, n])
                    self.weights += self.learning_rate * e @ patterns[:, n].reshape(1, -1)

            # stats
            y = self.predict(patterns)
            self.mce.append(len(np.where(targets != y)[0]) / patterns.shape[1])

    def predict(self, patterns):
        return self._activate(self.weights @ patterns)

    def _activate(self, x):
        if self.delta:
            targets = (1, -1)
        else:
            targets = (1, 0)
        return np.where(x > 0, targets[0], targets[1])

    _sigma = 0.01   # standard deviation for weight initialization


class TwoLayerPerceptron:
    """Two-layer perceptron (neural network with two layers of McCulloch Pitts neurons)."""

    def __init__(self, learning_rate, epochs, hidden_nodes, alpha):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_nodes = hidden_nodes    # number of neurons in the hidden layer
        self.alpha = alpha                  # factor for momentum term
        self.W = None                       # weights first layer
        self.V = None                       # weights second layer

    def learn(self, patterns, targets):
        """Train the perceptron using BackProp with momentum."""
        # init
        patterns = np.concatenate((patterns, np.ones(patterns.shape[1])), axis=0)           # add row for bias term
        self.W = self._sigma * np.random.randn(self.hidden_nodes, patterns.shape[0])
        self.V = self._sigma * np.random.randn(targets.shape[0], self.hidden_nodes+1)       # +1 for bias weight
        dW = 0
        dV = 0

        for i in range(self.epochs):
            # forward pass
            H = self._activate(self.W @ patterns)
            H = np.concatenate((H, np.ones(patterns.shape[0])), axis=0)                     # add row for bias term
            O = self._activate(self.V @ H)

            # backward pass
            delta_o = (O - targets) * self._derivative_activate(O)
            delta_h = (self.V.T @ delta_o) * self._derivative_activate(H)
            delta_h = delta_h[:-1]                                                          # remove extra row

            # weight update
            dW = self.alpha * dW - (1 - self.alpha) * (delta_o @ patterns.T)
            dV = self.alpha * dV - (1-self.alpha) * (delta_h @ H.T)
            self.W += self.learning_rate * dW
            self.V += self.learning_rate * dV

    def _activate(self, x):
        return 2. / (1 + np.exp(x)) - 1

    def _derivative_activate(self, phi_x):
        return (1 + phi_x) * (1 - phi_x) / 2

    _sigma = 0.01   # standard deviation for weight initialization
