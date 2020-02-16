import numpy as np
import matplotlib.pyplot as plt



class Hebbian_Network():

    def __init__(self):

        self.weights = None
        self.trained = False

    def little_model(self, data):

        P = data.shape[0]
        N = data.shape[1]
        normalizer = 1/N
        if not self.trained:
            self.weights = np.zeros((N,N))

        for pattern in range(P):
            self.weights += normalizer*np.outer(np.transpose(data[pattern,:]), data[pattern,:])
        self.trained = True

    def synch_update(self, data):

        new = np.dot(data, self.weights)
        result = np.where(new >= 0, 1, -1)
        return result

    def asynch_update(self, x):

        dim = np.shape(x)[0]
        new = x.copy()

        for unit in range(dim):
            weight_sum = 0
            i = np.random.randint(0,dim)
            for j in range(dim):

                weight_sum += np.multiply(self.weights[i][j], new[j])

            new[i] = np.sign(weight_sum)

        #plot_new = new.reshape(32, 32)
        #plt.imshow(plot_new)
        #plt.show()

        return new


    def recall(self, x, check_convergence=False, max_iters=None, method=None):

        error = 1

        iterations = 0
        energy = []

        while error > 0:

            if method == "synch":
                new = self.synch_update(x)
            elif method == "asynch":
                new = self.asynch_update(x)
            else:
                raise ValueError("Incorrectly specified method")
            error = np.sum(np.abs(new-x))
            x = new.copy()
            energy.append(self.energy(x))
            iterations += 1

            if iterations == max_iters:
                break
        print(iterations)

        return x, iterations, energy

    def energy(self, x):

        product = np.linalg.multi_dot([x, self.weights, x])
        energy = -np.sum(product)
        return energy


    def get_weights(self):


        return self.weights









