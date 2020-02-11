import numpy as np
import matplotlib.pyplot as plt
from RBFNN import RBFNN
from RBFNN import gaussian_rbf


def threshold(x):
    return np.where(x >= 0, 1., -1.)


def square(x):
    return threshold(np.sin(x))


def plot_rbf(rbf_nn, xlim):
    x = np.linspace(xlim[0], xlim[1])
    for i in range(rbf_nn.n_rbf):
        plt.plot(x_train, gaussian_rbf(x, rbf_nn.mu[i], rbf_nn.sigma))


xlim = 0, 2*np.pi
x_train = np.arange(xlim[0], xlim[1], 0.1).reshape(-1, 1)
x_test = np.arange(xlim[0], xlim[1], 0.1).reshape(-1, 1)
functions = [lambda x: np.sin(2*x), lambda x: square(2*x)]
noise = True
for i in range(len(functions)):
    f = functions[i]

    # generate targets
    f_train = f(x_train)
    f_test = f(x_test)
    t_train = f_train.copy()
    t_test = f_test.copy()
    if noise:
        sigma = 0.1
        t_train += np.random.normal(scale=sigma, size=f_train.shape)
        t_test += np.random.normal(scale=sigma, size=f_test.shape)

    # train
    n_rbf = 10
    sigma = 1
    mu = np.linspace(np.pi/2., 2*np.pi, n_rbf)
    rbf_nn = RBFNN(n_rbf=n_rbf, sigma=sigma, mu=mu)
    rbf_nn.learn(x_train, t_train)

    # test
    print('Residual error:', np.mean(abs(f_test - rbf_nn.predict(x_test))))
    plt.plot(x_train, rbf_nn.predict(x_train), 'b', label='approximate')
    plt.plot(x_train, f_train, 'k', label='true')
    plt.plot(x_train, t_train, 'or', label='training patterns')
    # plot_rbf(rbf_nn, xlim)
    if i == 1:  # square function
        print('Residual error (thresholded):', np.mean(abs(f_test - threshold(rbf_nn.predict(x_test)))))
        plt.plot(x_train, threshold(rbf_nn.predict(x_train)), 'y', label='approximate (thresholded)')
    plt.legend(loc='upper right')
    plt.show()
    print()
