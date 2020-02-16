import numpy as np
import matplotlib.pyplot as plt
from hopfieldnetwork import HopfieldNetwork


def load_pictures():
    return np.loadtxt('datasets/pict.dat', delimiter=",").reshape(-1, 1024)


def store_pictures(patterns):
    hn = HopfieldNetwork()
    hn.learn(patterns)
    if not check_stability(hn, patterns):
        print('Error, patterns not stored.')
        exit(-1)
    return hn


def check_stability(hn, patterns):
    for pattern in patterns:
        result = hn.recall(pattern, update_rule='synch', max_iters=1)
        if not result[2] or any(result[0] != pattern):
            return False
    return True


def plot_image(pattern, title=None):
    pattern = pattern.reshape(32, 32)
    plt.imshow(pattern)
    if title is not None:
        plt.title(title)
    plt.show()


def plot_energy(energy, title=None):
    plt.plot(energy)
    if title is not None:
        plt.title(title)
    plt.xlabel('iteration')
    plt.ylabel('energy')
    plt.xlim([0, len(energy)])
    plt.show()
