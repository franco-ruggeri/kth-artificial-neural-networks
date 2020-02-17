import numpy as np
import matplotlib.pyplot as plt
from hopfieldnetwork import HopfieldNetwork


def load_pictures():
    return np.loadtxt('datasets/pict.dat', delimiter=",").reshape(-1, 1024)


def store_pictures(pictures):
    hn = HopfieldNetwork()
    hn.learn(pictures)
    if not check_stability(hn, pictures):
        print('Error, patterns not stored.')
        exit(-1)
    return hn


def check_stability(hn, patterns):
    for pattern in patterns:
        result = hn.recall(pattern, update_rule='synch', max_iters=1)
        if not result[2] or any(result[0] != pattern):
            return False
    return True


def plot_picture(picture, title=None):
    picture = picture.reshape(32, 32)
    plt.imshow(picture)
    if title is not None:
        plt.title(title)
    plt.show()


def plot_energy(energy, label=None):
    if label is not None:
        plt.plot(energy, label=label)
        plt.legend()
    else:
        plt.plot(energy)
    plt.xlabel('iteration')
    plt.ylabel('energy')
    plt.xlim([0, len(energy)])


def add_noise(pattern, percentage):
    noisy_pattern = pattern.copy()
    shuffle_size = int(percentage * len(pattern))
    positions = np.random.choice(pattern.shape[0], shuffle_size, replace=False)
    old_values = noisy_pattern[positions]
    new_values = np.where(old_values == 1, -1, 1)
    noisy_pattern[positions] = new_values
    return noisy_pattern
