import numpy as np


class SLP:
    """
    Single-Layer Perceptron.

    Learning algorithm: perceptron or delta rule.
    Modes: batch or sequential.
    """

    def __init__(self, learning_rate, n_epochs, rule, mode, bias=True, init_weights=None, animation=False, seed=None):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.bias = bias
        self.weights = None
        self.seed = seed
        self._init_weights = init_weights
        self.animation = animation          # animation of decision boundary
        self.weights_history = []           # save history of weighs to realize animation of decision boundary
        self.mce = None                     # misclassification error

        # select learning rule
        if rule == 'delta':
            self.delta = True
        elif rule == 'perceptron':
            self.delta = False
        else:
            raise ValueError('Unknown rule, must be delta or perceptron.')

        # select mode
        if mode == 'batch':
            self.batch = True
        elif mode == 'sequential':
            self.batch = False
        else:
            raise ValueError('Unknown mode, must be batch or sequential.')

    def learn(self, patterns, targets):
        """Learn weights."""
        if self.seed is not None:
            np.random.seed(self.seed)

        # add extra row for bias term
        if self.bias:
            patterns = np.concatenate((patterns, np.ones((1, patterns.shape[1]))), axis=0)

        # init weights
        if self._init_weights is not None:
            self.weights = self._init_weights.copy()
        else:
            self.weights = self._sigma * np.random.randn(targets.shape[0], patterns.shape[0])

        # adjust labels
        if self.delta:
            targets = np.where(targets == 1, 1, -1)
        else:
            targets = np.where(targets == 1, 1, 0)

        # animation of decision boundary
        if self.animation:
            self.weights_history.append(self.weights.copy())

        # stats
        self.mce = [1 - self.score(patterns, targets)]

        for i in range(self.n_epochs):
            # update weights
            if self.batch:
                if self.delta:
                    e = targets - self.weights @ patterns
                else:
                    e = targets - self.predict(patterns)
                self.weights += self.learning_rate * e @ patterns.T
            else:
                for n in range(patterns.shape[1]):
                    pattern = patterns[:, n].reshape(-1, 1)
                    if self.delta:
                        e = targets[:, n] - self.weights @ pattern
                    else:
                        e = targets[:, n] - self.predict(pattern)
                    self.weights += self.learning_rate * e @ pattern.T

            # animation of decision boundary
            if self.animation:
                self.weights_history.append(self.weights.copy())

            # stats
            self.mce.append(1 - self.score(patterns, targets))

    def predict(self, patterns):
        """Predict targets."""
        # if used from outside, patterns are without extra row for bias term
        if patterns.shape[0] != self.weights.shape[1]:
            patterns = np.concatenate((patterns, np.ones((1, patterns.shape[1]))), axis=0)
        return self._activate(self.weights @ patterns)

    def score(self, patterns, targets):
        """Compute accuracy."""
        # adjust labels
        if self.delta:
            targets = np.where(targets == 1, 1, -1)
        else:
            targets = np.where(targets == 1, 1, 0)

        # score
        y = self.predict(patterns)
        correct = np.all(np.equal(y, targets), axis=0)
        n_correct = len(np.where(correct)[0])
        return n_correct / patterns.shape[1]

    def _activate(self, x):
        """Compute activation of neurons."""
        if self.delta:
            labels = (1, -1)
        else:
            labels = (1, 0)
        return np.where(x >= 0, labels[0], labels[1])

    _sigma = 0.01   # standard deviation for weight initialization


class MLP:
    """
    Multi-Layer perceptron.

    Learning algorithm: BackProp (generalized delta rule) with momentum.
    Mode: batch.
    """

    def __init__(self, learning_rate, n_epochs, n_hidden, mode, alpha=0.9, seed=None):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_hidden = n_hidden            # number of neurons in the hidden layer
        self.alpha = alpha                  # factor for momentum term
        self.W = None                       # weights first layer
        self.V = None                       # weights second layer
        self._dW = None                      # previous update for W, used for momentum
        self._dV = None                      # previous update for V, used for momentum
        self.mce = None                     # misclassification error on the training set
        self.mce_val = None                 # misclassification error on the validation set
        self.seed = seed

        # select mode
        if mode == 'batch':
            self.batch = True
        elif mode == 'sequential':
            self.batch = False
        else:
            raise ValueError('Unknown mode, must be batch or sequential.')

    def learn(self, patterns, targets, val_patterns=None, val_targets=None):
        """Learn weights."""
        if self.seed is not None:
            np.random.seed(self.seed)

        # extra row for bias term
        patterns = np.concatenate((patterns, np.ones((1, patterns.shape[1]))), axis=0)

        # tip 1 - smart initial weights (uniform distribution with zero-mean and variance N => U(-sqrt(3*N), sqrt(3*N)))
        low = -np.sqrt(3*patterns.shape[0])
        high = -low
        self.W = np.random.uniform(low, high, size=(self.n_hidden, patterns.shape[0]))
        low = -np.sqrt(3*(self.n_hidden+1))
        high = -low
        self.V = np.random.uniform(low, high, size=(targets.shape[0], self.n_hidden + 1))

        # stats
        self.mce = [1 - self.score(patterns, targets)]
        if val_patterns is not None and val_targets is not None:
            validate = True
            self.mce_val = [1 - self.score(val_patterns, val_targets)]
        else:
            validate = False

        self._dW = 0
        self._dV = 0
        for i in range(self.n_epochs):
            if self.batch:
                self._backprop(patterns, targets)
            else:
                # tip 2 - shuffle at each epoch (improve stochastic behavior)
                idx = np.arange(patterns.shape[1])
                np.random.shuffle(idx)
                patterns_ = patterns[:, idx]
                targets_ = targets[:, idx]

                for n in range(patterns_.shape[1]):
                    pattern = patterns_[:, n].reshape(-1, 1)
                    target = targets_[:, n].reshape(-1, 1)
                    self._backprop(pattern, target)

            # stats
            self.mce.append(1 - self.score(patterns, targets))
            if validate:
                self.mce_val.append(1 - self.score(val_patterns, val_targets))

    def predict(self, patterns, thresholded=True):
        """Predict targets."""
        # if used from outside, patterns are without extra row for bias term
        if patterns.shape[0] != self.W.shape[1]:
            patterns = np.concatenate((patterns, np.ones((1, patterns.shape[1]))), axis=0)
        O = self._forward(patterns)[1]
        if thresholded:
            O = np.where(O >= 0, 1, -1)
        return O

    def score(self, patterns, targets):
        """Compute accuracy."""
        y = self.predict(patterns)
        correct = np.all(np.equal(y, targets), axis=0)
        n_correct = len(np.where(correct)[0])
        return n_correct / patterns.shape[1]

    def _forward(self, patterns):
        """Forward pass of BackProp."""
        H = self._activate(self.W @ patterns)
        H = np.concatenate((H, np.ones((1, patterns.shape[1]))), axis=0)  # add row for bias term
        O = self._activate(self.V @ H)
        return H, O

    def _backprop(self, patterns, targets):
        """One iteration of BackProp."""
        # forward pass
        H, O = self._forward(patterns)

        # backward pass
        delta_o = (O - targets) * self._derivative_activate(O)
        delta_h = (self.V.T @ delta_o) * self._derivative_activate(H)
        delta_h = delta_h[:-1]  # remove extra row

        # weight update
        self._dW = self.alpha * self._dW - (1 - self.alpha) * (delta_h @ patterns.T)
        self._dV = self.alpha * self._dV - (1 - self.alpha) * (delta_o @ H.T)
        self.W += self.learning_rate * self._dW
        self.V += self.learning_rate * self._dV

    def _activate(self, x):
        """Compute activation of neurons."""
        return 2. / (1 + np.exp(-x)) - 1

    def _derivative_activate(self, phi_x):
        """Compute derivative of activation of neurons, given the activation (necessary for backward pass)."""
        return (1 + phi_x) * (1 - phi_x) / 2
