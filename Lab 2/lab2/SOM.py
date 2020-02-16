import numpy as np
import math


def manhattan(x, y):
    """Manhattan distance."""
    return abs(x[0] - y[0]) + abs(x[1] - y[1])


def _neighbourhood_1d(winner, n_nodes, neighbourhood_size, circular):
    """Get neighbourhood, represented as indexes, of a node. The node itself is included. 1D version."""
    idx_min, idx_max = math.ceil(winner - neighbourhood_size/2.), math.floor(winner + neighbourhood_size/2.)
    if circular:
        if idx_max >= n_nodes:
            neighbourhood = np.arange(idx_min, n_nodes)  # [idx_min, n_nodes-1] plus...
            idx_max %= n_nodes
            neighbourhood = np.concatenate((neighbourhood, np.arange(idx_max + 1)))  # ... [0, idx_max % n_nodes]
        else:
            neighbourhood = np.arange(idx_min, idx_max + 1)
    else:
        if idx_min < 0:
            idx_min = 0
        if idx_max >= n_nodes:
            idx_max = n_nodes - 1
        neighbourhood = np.arange(idx_min, idx_max + 1)
    return neighbourhood


def _neighbourhood_2d(winner, side, neighbourhood_size):
<<<<<<< HEAD
    """Get neighbourhood, represented as indexes, of a node. The node itself is included. 2D version."""
    distances = [manhattan(i, j) for j in range(side) for i in range(side)]
    distances = sorted(distances)
    return np.array(distances[:neighbourhood_size])
=======
    """
    Get neighbourhood, represented as indexes, of a node. The node itself is included. 2D version.
    The neighbourhood size represents the "radius" of the neighbourhood, not the total size ("area").
    """
    return np.array([(i, j) for j in range(side) for i in range(side)
                     if manhattan(winner, (i, j)) <= neighbourhood_size])
>>>>>>> c8abaae2470f708406423fb6613e12dbb0b2b7d2


def _select_distance(distance):
    """Select distance function for pattern and nodes (input space)."""
    if distance == 'euclidean':
        return lambda x, y: (x - y).dot(x - y)  # no square root, not necessary


def _select_neighbourhood(dim, n_nodes, circular):
    """Select neighbourhood function depending on the dimensionality of the output space."""
    if dim == 1:
        return lambda winner, neighbourhood_size: _neighbourhood_1d(winner, n_nodes, neighbourhood_size, circular)
    elif dim == 2:
        return lambda winner, neighbourhood_size: _neighbourhood_2d(winner, int(math.sqrt(n_nodes)), neighbourhood_size)


class SOM:
    """Self-organizing map. Support for 1D or 2D grid of nodes."""

    def __init__(self, dim, n_nodes, learning_rate, n_epochs, init_nb_size, circular=False, distance='euclidean'):
        self.n_nodes = n_nodes
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self._init_nb_size = init_nb_size  # initial neighbourhood size
        self.dim = dim
        self.circular = circular
        self.weights = None
        self._distance = _select_distance(distance)
        self._neighbourhood = _select_neighbourhood(dim, n_nodes, circular)

    def learn(self, patterns):
        """Learn map from patterns."""
        self._init_weights(patterns)
        for i in range(self.n_epochs):
            for p in patterns:
                winner = self.winner(p)
                neighbourhood_size = round(self._init_nb_size / self.n_epochs * (self.n_epochs - i))
                neighbourhood = self._neighbourhood(winner, neighbourhood_size)
                self._update_weights(p, neighbourhood)

                # debug
                # print('Epoch:', i)
                # print('Winner:', winner)
                # print('Neighbourhood size:', neighbourhood_size)
                # print('Actual neighbourhood size:', len(neighbourhood))
                # print()

    def winner(self, pattern):
        """Get the index of the winner node."""
        if self.dim == 1:
            distances = [self._distance(pattern, w) for w in self.weights]
            winner = distances.index(min(distances))
        elif self.dim == 2:
            distances = np.array([[self._distance(pattern, self.weights[i, j]) for j in range(self.weights.shape[1])]
                                  for i in range(self.weights.shape[0])])
            winner = np.where(distances == np.amin(distances))
            winner = (winner[0][0], winner[1][0])   # take the first one
        return winner

    def _update_weights(self, pattern, neighbourhood):
        """Move nodes in the neighbourhood a bit closer to the pattern."""
        if self.dim == 1:
            self.weights[neighbourhood] += self.learning_rate * (pattern - self.weights[neighbourhood])
        elif self.dim == 2:
            self.weights[neighbourhood[:, 0], neighbourhood[:, 1]] += \
                self.learning_rate * (pattern - self.weights[neighbourhood[:, 0], neighbourhood[:, 1]])

    def _init_weights(self, patterns):
        """Init the weight grid depending on the dimensionality of the output space."""
        if self.dim == 1:
            self.weights = np.random.rand(self.n_nodes, patterns.shape[1])
        elif self.dim == 2:
            side = int(math.sqrt(self.n_nodes))
            self.weights = np.random.rand(side, side, patterns.shape[1])
