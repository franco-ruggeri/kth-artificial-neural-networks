import numpy as np


class SLP:
    """Single-Layer Perceptron (neural network with one layer of McCulloch Pitts neurons)."""

    def __init__(self, learning_rate, epochs, method, mode, bias=True, animation=False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.bias = bias
        self.weights = None
        self.mce = None                 # misclassification error
        self.animation = animation      # plot decision boundary at each epoch
        self.mse = None

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
        self.mce = [len(np.where(targets != y)[0]) / patterns.shape[1]]

        for i in range(self.epochs):
            # animation of decision boundary
            if self.animation:
                xlim = [min(patterns[0]) - 0.1, max(patterns[0]) + 0.1, min(patterns[1]) - 0.1, max(patterns[1]) + 0.1]
                self.plot_decision_boundary(xlim, 'y')

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
        if patterns.shape[0] != self.weights.shape[1]:  # used from outside, patterns without extra row for bias
            patterns = np.concatenate((patterns, np.ones((1, patterns.shape[1]))), axis=0)
        return self._activate(self.weights @ patterns)

    def _activate(self, x):
        if self.delta:
            targets = (1, -1)
        else:
            targets = (1, 0)
        return np.where(x > 0, targets[0], targets[1])

    _sigma = 0.01   # standard deviation for weight initialization


class MLP:
    """Two-layer perceptron (neural network with two layers of McCulloch Pitts neurons)."""

    def __init__(self, learning_rate, epochs, hidden_nodes, alpha=0.9):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_nodes = hidden_nodes    # number of neurons in the hidden layer
        self.alpha = alpha                  # factor for momentum term
        self.W = None                       # weights first layer
        self.V = None                       # weights second layer
        self.mce = None                     # misclassification error
        self.val_mce = None
        self.mse_over_time = []
        self.val_mse_over_time = []

    def learn(self, patterns, targets, validation_patterns, validation_targets):
        """Train the perceptron using BackProp with momentum."""
        # init (see Marsland's book for initialization of weights)
        patterns = np.concatenate((patterns, np.ones((1, patterns.shape[1]))), axis=0)
        self.W = 1. / np.sqrt(patterns.shape[0]) * (np.random.rand(self.hidden_nodes, patterns.shape[0]) - 0.5)
        self.V = 1. / np.sqrt(self.hidden_nodes+1) * (np.random.rand(targets.shape[0], self.hidden_nodes+1) - 0.5)
        dW = 0
        dV = 0

        # stats
        y = self.predict(patterns, classification=True)
        self.mce = [len(np.where(targets != y)[0]) / patterns.shape[1]]

        if validation_patterns is not None:
            y_val = self.predict(validation_patterns, classification=True)
            self.val_mce = [len(np.where(validation_targets != y_val)[0]) / validation_patterns.shape[1]]

        # for auto encoder part
        #converged = False
        epoch = 0
        #while not converged:

        # for all the other parts
        for i in range(self.epochs):
            # forward pass
            H, O = self._forward(patterns)

            # backward pass
            delta_o = (O - targets) * self._derivative_activate(O)
            delta_h = (self.V.T @ delta_o) * self._derivative_activate(H)
            delta_h = delta_h[:-1]                                                      # remove extra row

            # weight update
            dW = self.alpha * dW - (1 - self.alpha) * (delta_h @ patterns.T)
            dV = self.alpha * dV - (1-self.alpha) * (delta_o @ H.T)
            self.W += self.learning_rate * dW
            self.V += self.learning_rate * dV

            # stats
            y = self.predict(patterns, classification=True)
            self.mce.append(len(np.where(targets != y)[0]) / patterns.shape[1])

            if validation_patterns is not None:
                y_val = self.predict(validation_patterns, classification=True)
                self.val_mce.append(len(np.where(validation_targets != y_val)[0]) / validation_patterns.shape[1])

                self.val_mse_over_time.append(
                    np.sum(np.square(self.predict(validation_patterns, classification=False) - validation_targets))
                    / np.shape(validation_targets)[1])

            if np.array_equal(np.around(O), targets):
                return epoch, np.where(H > 0, 1, 0)

            epoch += 1

            self.mse_over_time.append(np.sum(np.square(self.predict(patterns, classification=False) - targets))
                                      / np.shape(targets)[1])



    def predict(self, patterns, classification=False):
        if patterns.shape[0] != self.W.shape[1]:    # used from outside, patterns without extra row for bias
            patterns = np.concatenate((patterns, np.ones((1, patterns.shape[1]))), axis=0)

        O = self._forward(patterns)[1]
        if classification:
            O = np.where(abs(O) < 1e-6, 0, O)           # on the decision boundary
            O = np.where(O > 0, 1, O)
            O = np.where(O < 0, -1, O)
            return O
        else:
            return O

    def _forward(self, patterns):
        H = self._activate(self.W @ patterns)
        H = np.concatenate((H, np.ones((1, patterns.shape[1]))), axis=0)                # add row for bias term
        O = self._activate(self.V @ H)
        return H, O

    def _activate(self, x):
        return 2. / (1 + np.exp(-x)) - 1

    def _derivative_activate(self, phi_x):
        return (1 + phi_x) * (1 - phi_x) / 2
