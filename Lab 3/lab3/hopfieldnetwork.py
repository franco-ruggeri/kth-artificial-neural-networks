import numpy as np


def sign(x):
    return np.where(x >= 0, 1, -1)


def compute_activity(patterns):
    N, P = patterns.shape
    return 1. / (N*P) * np.sum(patterns)


class HopfieldNetwork:
    """Hopfield Neural Network using Hebbian learning."""

    def __init__(self, self_connections=False, sparse=False, bias=None):
        self.weights = None
        self.n_neurons = None
        self.trained = False
        self.self_connections = self_connections
        self.sparse = sparse
        self.bias = bias

    def learn(self, patterns):
        """Hebbian learning."""
        self.n_neurons = patterns.shape[1]
        if not self.trained:
            self.weights = np.zeros((self.n_neurons, self.n_neurons))

        if self.sparse:
            patterns = patterns - compute_activity(patterns)
            normalizer = 1
        else:
            normalizer = 1 / self.n_neurons

        for pattern in patterns:
            self.weights += normalizer*np.outer(pattern, pattern)
        if not self.self_connections:
            np.fill_diagonal(self.weights, 0)     # no self connections
        self.trained = True

    def recall(self, pattern, update_rule='asynch', max_iters=None, plot=False):
        """Feed pattern into the network to recall a stored pattern."""
        error = 1
        iterations = -1
        energy = []
        state = pattern.copy()

        while error > 0:
            iterations += 1
            if iterations == max_iters:
                return state, energy, False  # not converged

            if update_rule == 'synch':
                new_state, energy_ = self._synchronous_update(state)
                energy.append(energy_)
            elif update_rule == 'asynch':
                new_state, energy_ = self._asynchronous_update(state, plot)
                energy.extend(energy_)
            else:
                raise ValueError("Wrong update rule specified")

            error = np.sum(np.abs(new_state - state))
            state = new_state
            # print("Iteration:", iterations+1)
        return state, energy, True

    def _synchronous_update(self, state):
        """Update weights synchronously."""
        if not self.sparse:
            state = sign(self.weights.dot(state))
        else:
            state = 0.5 + 0.5 * sign(self.weights.dot(state) - self.bias)
        energy = self._compute_energy(state)
        return state, energy

    def _asynchronous_update(self, state, plot=False):
        """Update weights asynchronously (sequentially)."""
        order = np.arange(self.n_neurons)
        np.random.shuffle(order)
        state = state
        energy = []

        for i, idx in enumerate(order):
            if not self.sparse:
                state[idx] = sign(self.weights[idx].dot(state))
            else:
                state[idx] = 0.5 + 0.5 * sign(self.weights[idx].dot(state) - self.bias)
            energy.append(self._compute_energy(state))

            # plot each 100 iterations
            if plot and i % 100 == 0:
                import utility as u
                u.plot_picture(state, title='Sequential dynamics - {} iterations'.format(i))
        return state, energy

    def _compute_energy(self, state):
        """Compute energy (Lyapunov function)."""
        product = np.linalg.multi_dot([state, self.weights, state])
        energy = -np.sum(product)
        return energy
