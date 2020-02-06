import numpy as np
import math


def _select_distance(distance):
    if distance == 'euclidean':
        return lambda x, y: (x-y).dot(x-y)  # no square root, not necessary


class SOM:
    """Self-organizing map with 1D arrangement of nodes."""

    def __init__(self, n_nodes, learning_rate, n_epochs, init_nb_size, circular=False, distance='euclidean',
                 verbose=False):
        self.n_nodes = n_nodes
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self._init_nb_size = init_nb_size  # initial neighbourhood size
        self._circular = circular
        self._distance = _select_distance(distance)
        self._verbose = verbose
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
        distances = [self._distance(pattern, w) for w in self.weights]
        winner = distances.index(min(distances))
        return winner

    def _neighbourhood(self, node, epoch):
        """Get neighbourhood, represented as indexes, of a node. The node itself is included."""
        neighbourhood_size = round(self._init_nb_size / self.n_epochs * (self.n_epochs - epoch))
        idx_min, idx_max = math.ceil(node - neighbourhood_size), math.floor(node + neighbourhood_size)
        if self._circular:
            if idx_max >= self.n_nodes:
                neighbourhood = np.arange(idx_min, self.n_nodes)                        # [idx_min, n_nodes-1] plus...
                idx_max %= self.n_nodes
                neighbourhood = np.concatenate((neighbourhood, np.arange(idx_max+1)))   # ... [0, idx_max % n_nodes]
            else:
                neighbourhood = np.arange(idx_min, idx_max + 1)
        else:
            if idx_min < 0:
                idx_min = 0
            if idx_max >= self.n_nodes:
                idx_max = self.n_nodes-1
            neighbourhood = np.arange(idx_min, idx_max + 1)
        if self._verbose:
            print('epoch:', epoch)
            print('winner:', node)
            print('neighbourhood size:', neighbourhood_size)
            print('range:', idx_min, idx_max)
            print('effective neighbourhood size:', len(neighbourhood))
            print('effective neighbourhood:', neighbourhood)
            print()
        return neighbourhood

    def _update_weights(self, pattern, neighbourhood):
        """Move nodes a bit closer to the pattern."""
        self.weights[neighbourhood] += self.learning_rate * (pattern - self.weights[neighbourhood])
