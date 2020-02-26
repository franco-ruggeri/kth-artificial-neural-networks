import numpy as np
from numpy.linalg import inv


def _select_distance(distance):
    """Select distance function for pattern and nodes (input space)."""
    if distance == 'euclidean':
        return lambda x, y: (x - y).dot(x - y)  # no square root, not necessary


def gaussian_rbf(x, mu, sigma):
    """Gaussian RBF."""
    return np.exp(-(x-mu)**2 / (2*sigma**2))


class RBFNN:
    """RBF neural network."""

    def __init__(self, n_rbf, sigma, mu=None, mode='batch', learning_rate=None, n_epochs=None, cl_iterations=None,
                 distance='euclidean'):
        self.n_rbf = n_rbf
        self.sigma = sigma
        self.mu = mu
        self.mode = mode
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.cl_iterations = cl_iterations
        self._distance = _select_distance(distance)
        self.weights = None

    def learn(self, patterns, targets):
        if self.mu is None:
            self._competitive_learning()

        phi = self._get_phi(patterns)
        if self.mode == 'batch':
            self.weights = inv(phi.T @ phi) @ phi.T @ targets
        elif self.mode == 'online':
            self.weights = np.random.rand(self.n_rbf, targets.shape[1])
            for i in range(self.n_epochs):
                idx = np.arange(len(patterns))
                np.random.shuffle(idx)
                for j in idx:
                    self.weights += self.learning_rate * \
                                (targets[j] - phi[j].reshape(1, -1) @ self.weights) * phi[j].reshape(-1, 1)

    def predict(self, patterns):
        phi = self._get_phi(patterns)
        return phi @ self.weights

    def _competitive_learning(self, patterns):
        # init with data samples
        idx = np.arange(len(patterns))
        np.shuffle(idx)
        self.mu = patterns[idx[:self.n_rbf]]

        for i in range(self.cl_iterations):
            np.shuffle(idx)
            pattern = patterns[idx[0]]
            winner = self._winner(pattern)
            self._update_rbf(winner)

    def _winner(self, pattern):
        distances = [self._distance(pattern, mu) for mu in self.mu]
        winner = distances.index(min(distances))
        return winner

    def _update_rbf(self, pattern, winner):
        self.mu[winner] += self.learning_rate * (pattern - self.mu[winner])

    def _phi(self, x):
        return np.array([gaussian_rbf(x, mu, self.sigma) for mu in self.mu]).reshape(-1, 1)

    def _get_phi(self, patterns):
        return np.array([self._phi(x).T for x in patterns]).reshape(-1, self.n_rbf)
