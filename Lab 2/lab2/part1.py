import numpy as np
import matplotlib.pyplot as plt
from RBFNN import RBFNN
from RBFNN import gaussian_rbf


def threshold(x):
    return np.where(x >= 0, 1, -1)


def square(x):
    return threshold(np.sin(x))


def plot_rbf(rbf_nn, x):
    for i in range(rbf_nn.n_rbf):
        plt.plot(x_train, gaussian_rbf(x_train, rbf_nn.mu[i], rbf_nn.sigma))


start, stop = 0, 2*np.pi
x_train = np.arange(start, stop, 0.1).reshape(-1, 1)
x_test = np.arange(start, stop, 0.1).reshape(-1, 1)
for f in [lambda x: np.sin(2*x), lambda x: square(2*x)]:
    # generate targets
    f_train = f(x_train)
    f_test = f(x_test)

    # train
    n_rbf = 10
    sigma = 1
    mu = np.linspace(np.pi/2., 2*np.pi, n_rbf)
    rbf_nn = RBFNN(n_rbf=n_rbf, sigma=sigma, mu=mu)
    rbf_nn.learn(x_train, f_train)

    # test
    print('Residual error:', np.mean(abs(f_test - rbf_nn.predict(x_test))))
    print('Residual error (thresholded):', np.mean(abs(f_test - threshold(rbf_nn.predict(x_test)))))
    plt.plot(x_train, rbf_nn.predict(x_train), label='approximate')
    plt.plot(x_train, f_train, label='true')
    plt.plot(x_train, threshold(rbf_nn.predict(x_train)), label='approximate (thresholded)')
    # plot_rbf(rbf_nn, x_train)
    plt.legend()
    plt.show()
