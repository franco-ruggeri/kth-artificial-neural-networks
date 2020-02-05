import numpy as np


def _select_distance(distance):
    if distance == 'euclidean':
        return lambda x, y: (x-y).dot(x-y)  # no square root, not necessary


class SOM:
    """Self-organizing map with 1D map."""

    def __init__(self, n_nodes, learning_rate, n_epochs, distance='euclidean', init_nb_size=50):
        self.n_nodes = n_nodes
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self._distance = _select_distance(distance)
        self._init_nb_size = init_nb_size   # initial neighbourhood size
        self.weights = None

    def learn(self, patterns):
        """Learn map from patterns."""
        self.weights = np.random.rand(self.n_nodes, patterns.shape[1])
        for i in range(self.n_epochs):
            for p in patterns:
                winner = self.winner(p)
                neighbourhood = self._neighbourhood(winner, i)
                self._update_weights(p, neighbourhood)

    def winner(self, pattern):
        """Get the index of the winner node."""
        distances = list(map(lambda w: self._distance(pattern, w), weights))
        winner = distances.index(min(distances))
        return winner

    def _neighbourhood(self, node, epoch):
        """Get neighbourhood, represented as indexes, of a node. The node itself is included."""
        neighbourhood_size = int(self._init_nb_size / self.n_epochs * epoch)
        return np.arange(node-neighbourhood_size, node+neighbourhood_size)

    def _update_weights(self, pattern, neighbourhood):
        """Move nodes a bit closer to the pattern."""
        for i in neighbourhood:
            self.weights[i] += self.learning_rate * (pattern - self.weights[i])
