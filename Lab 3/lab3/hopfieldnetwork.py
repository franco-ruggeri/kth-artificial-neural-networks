import numpy as np


def sign(x):
    return np.where(x >= 0, 1, -1)


class HopfieldNetwork:
    """Hopfield Neural Network using Hebbian learning."""

    def __init__(self):
        self.weights = None
        self.trained = False

    def learn(self, patterns):
        """Hebbian learning."""
        N = patterns.shape[1]
        normalizer = 1/N
        if not self.trained:
            self.weights = np.zeros((N, N))

        for pattern in patterns:
            self.weights += normalizer*np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)     # no self connections
        self.trained = True

    def recall(self, pattern, check_convergence=False, max_iters=None):
        """Feed pattern into the network to recall a stored pattern."""
        error = 1
        iterations = -1
        energy = []

        state = pattern
        while error > 0:
            iterations += 1
            if iterations == max_iters:
                return None, energy  # not converged

            new_state = self.synch_update(state)
            error = np.sum(np.abs(new_state - state))
            state = new_state
            energy.append(self.energy(state))
            if check_convergence:
                return state, iterations, energy
        return state, energy

    def synch_update(self, pattern):
        return sign(self.weights.dot(pattern))

    def asynch_update(self, x):
        dim = np.shape(x)[0]
        new = x.copy()

        for unit in range(dim):
            weight_sum = 0
            i = np.random.randint(0,dim)
            for j in range(dim):

                weight_sum += np.multiply(self.weights[i][j], new[j])

            new[i] = np.sign(weight_sum)

        # plot_new = new.reshape(32, 32)
        # plt.imshow(plot_new)
        # plt.show()

        return new

    def energy(self, x):
        product = np.linalg.multi_dot([x, self.weights, x])
        energy = -np.sum(product)
        return energy
