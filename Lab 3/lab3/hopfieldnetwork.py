import numpy as np
import utility as u


def sign(x):
    return np.where(x >= 0, 1, -1)


class HopfieldNetwork:
    """Hopfield Neural Network using Hebbian learning."""

    def __init__(self):
        self.weights = None
        self.n_neurons = None
        self.trained = False

    def learn(self, patterns):
        """Hebbian learning."""
        self.n_neurons = patterns.shape[1]
        if not self.trained:
            self.weights = np.zeros((self.n_neurons, self.n_neurons))
        normalizer = 1 / self.n_neurons

        for pattern in patterns:
            self.weights += normalizer*np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)     # no self connections
        self.trained = True

    def recall(self, pattern, synchronous=False, max_iters=None, plot=False):
        """Feed pattern into the network to recall a stored pattern."""
        error = 1
        iterations = -1
        energy = []
        state = pattern.copy()

        while error > 0:
            iterations += 1
            if iterations == max_iters:
                return state, energy, False  # not converged

            if synchronous:
                new_state = self._synchronous_update(state)
                energy.append(self.compute_energy(new_state))
            else:
                new_state, energy_ = self._asynchronous_update(state, plot)
                energy.extend(energy_)

            error = np.sum(np.abs(new_state - state))
            state = new_state
        return state, energy, True

    def _synchronous_update(self, state):
        """Update weights synchronously."""
        return sign(self.weights.dot(state))

    def _asynchronous_update(self, state, plot=False):
        """Update weights asynchronous (sequentially)."""
        order = np.arange(self.n_neurons)
        np.random.shuffle(order)
        state = state
        energy = []
        i = 0

        for idx in order:
            state[idx] = sign(self.weights[idx].dot(state))
            energy.append(self.compute_energy(state))

            # plot each 100 iterations
            i += 1
            if plot and i % 100 == 0:
                u.plot_image(state, title='Sequential dynamics - {} iterations'.format(i))
        return state, energy

    def compute_energy(self, state):
        """Compute energy (Lyapunov function)."""
        product = np.linalg.multi_dot([state, self.weights, state])
        energy = -np.sum(product)
        return energy
