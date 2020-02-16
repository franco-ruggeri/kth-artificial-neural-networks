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

    def learn(self, patterns, sparse = False, activity = None, remove_self_connections = False):
        """Hebbian learning."""
        self.n_neurons = patterns.shape[1]
        if not self.trained:
            self.weights = np.zeros((self.n_neurons, self.n_neurons))
        normalizer = 1 / self.n_neurons

        if sparse:
            if activity == None:
                raise ValueError("Activity level not set")

            activity = np.ones(patterns.shape)*activity
            patterns = patterns - activity

        for pattern in patterns:
            self.weights += normalizer*np.outer(pattern, pattern)
        if remove_self_connections:
            np.fill_diagonal(self.weights, 0)     # no self connections
        self.trained = True

    def recall(self, pattern, update_rule=None, max_iters=None, theta=None, plot=False):
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
                new_state = self._synchronous_update(state)
                energy.append(self.compute_energy(new_state))

            elif update_rule == 'asynch':
                new_state, energy_ = self._asynchronous_update(state, plot)
                energy.extend(energy_)

            elif update_rule == 'sparse':
                new_state = self._sparse_update(state, theta)
                energy.append(self.compute_energy(new_state))

            else:
                raise ValueError("No update rule is specified")

            error = np.sum(np.abs(new_state - state))
            state = new_state
            #print("Error:", error)
            #print("Iteration:", iterations)
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

    def _sparse_update(self, state, theta):
        """Update the weights sparsely"""
        return 0.5 + 0.5*sign(self.weights.dot(state) - theta)

    def compute_energy(self, state):
        """Compute energy (Lyapunov function)."""
        product = np.linalg.multi_dot([state, self.weights, state])
        energy = -np.sum(product)
        return energy
