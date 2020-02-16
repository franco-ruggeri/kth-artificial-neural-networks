import numpy as np
import matplotlib.pyplot as plt
from RBFNN import RBFNN


def sign(x):
    return np.where(x >= 0, 1., -1.)


def square(x):
    return sign(np.sin(x))


def get_function(f):
    if f == 'sin':
        return lambda x: np.sin(2*x)
    elif f == 'square':
        return lambda x: square(2*x)


def generate_data(f, noise=False):
    xlim = 0, 2 * np.pi
    x_train = np.arange(xlim[0], xlim[1], 0.1).reshape(-1, 1)
    x_test = np.arange(xlim[0]+0.05, xlim[1]+0.05, 0.1).reshape(-1, 1)
    f_train = f(x_train)
    f_test = f(x_test)
    if noise:   # add noise
        sigma = 0.1
        f_train += np.random.normal(scale=sigma, size=f_train.shape)
        f_test += np.random.normal(scale=sigma, size=f_test.shape)
    return x_train, f_train, x_test, f_test


def plot_data(x_train, f):
    xlim = 0, 2*np.pi
    x = np.linspace(xlim[0], xlim[1])
    plt.plot(x, f(x), 'k', label='true function')
    plt.plot(x_train, f_train, 'oy', label='training pattern')


def init_rbf(n_rbf):
    # RBF centres uniformly distributed, but not from 0 because otherwise they would be biased to the left
    mu = np.arange(2 * np.pi / (2 * n_rbf), 2 * np.pi, 2 * np.pi / n_rbf)

    # RBF width is found according to Marsland's book: d/(2*n_rbf), where d is the maximum distance between the centres
    d = max([abs(mu[i] - mu[j]) for i in range(len(mu)) for j in range(len(mu)) if i > j])
    sigma = d / np.sqrt(2 * n_rbf)
    return mu, sigma


def train_rbf_nn(n_rbf, x_train, f_train, mode='batch', learning_rate=None, n_epochs=None, sigma=None):
    mu, aux = init_rbf(n_rbf)
    if sigma is None:
        sigma = aux
    rbf_nn = RBFNN(n_rbf=n_rbf, sigma=sigma, mu=mu, mode=mode, learning_rate=learning_rate, n_epochs=n_epochs)
    rbf_nn.learn(x_train, f_train)
    return rbf_nn


def plot_rbf_nn(rbf_nn, f, label, transform=False):
    xlim = 0, 2*np.pi
    x = np.linspace(xlim[0], xlim[1])
    f_predict = rbf_nn.predict(x)
    if transform:
        f_predict = sign(f_predict)
    plt.plot(x, f_predict, label=label)


def get_minimum_num_rbf_units(threshold, x_train, f_train, x_test, f_test):
    residual_error = np.Infinity
    n_rbf = 1
    while residual_error > threshold:
        n_rbf += 1

        rbf_nn = train_rbf_nn(n_rbf, x_train, f_train)
        f_predict = rbf_nn.predict(x_test)
        if transform:
            f_predict = sign(f_predict)
        residual_error = np.mean(abs(f_test - f_predict))
        print('Residual error:', residual_error)
    return n_rbf


np.random.seed(1)

f = 'sin'
f = get_function(f)
noise = True
x_train, f_train, x_test, f_test = generate_data(f, noise)
plot_data(x_train, f)

# 3.1
# threshold = 0.1
# transform = False
# n_rbf = get_minimum_num_rbf_units(threshold, x_train, f_train, x_test, f_test)
# print('Number of RBF units for threshold {}: {}'.format(threshold, n_rbf))
# rbf_nn = train_rbf_nn(n_rbf, x_train, f_train)
# plot_rbf_nn(rbf_nn, f, 'approximate function', transform)

# 3.2
# for sigma in [0.1, 10]:
#     for n_rbf in [2, 10]:
#         rbf_nn = train_rbf_nn(n_rbf, x_train, f_train, mode='online', learning_rate=0.1, n_epochs=100, sigma=sigma)
#         plot_rbf_nn(rbf_nn, f, r'online n_rbf={} $\sigma=${}'.format(n_rbf, sigma))
#         rbf_nn = train_rbf_nn(n_rbf, x_train, f_train, mode='batch', sigma=sigma)
#         plot_rbf_nn(rbf_nn, f, r'batch n_rbf={} $\sigma=${}'.format(n_rbf, sigma))

# 3.3
# TODO

plt.legend(loc='upper right')
plt.show()
