import numpy as np
from numpy.linalg import inv


def gaussian_rbf(x, mu, sigma):
    """Gaussian RBF."""
    return np.exp(-(x-mu)**2 / (2*sigma**2))


class RBFNN:
    """RBF neural network."""

    def __init__(self, n_rbf, sigma, mu, method, learning_rate=None, n_epochs=None):
        self.n_rbf = n_rbf
        self.sigma = sigma
        self.mu = mu
        self.method = method
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None

    def learn(self, patterns, targets):
        if self.method == 'batch':
            self.weights = np.random.rand(targets.shape[1], self.n_rbf)
            phi = self._get_phi(patterns)
            self.weights = inv(phi.T @ phi) @ phi.T @ targets
        elif self.method == 'online':
            for i in range(self.n_epochs):
                # TODO: implement
                pass

    def predict(self, patterns):
        phi = self._get_phi(patterns)
        return phi @ self.weights

    def _phi(self, x):
        return np.array([gaussian_rbf(x, mu, self.sigma) for mu in self.mu]).reshape(-1, 1)

    def _get_phi(self, patterns):
        return np.array([self._phi(x).T for x in patterns]).reshape(-1, self.n_rbf)
