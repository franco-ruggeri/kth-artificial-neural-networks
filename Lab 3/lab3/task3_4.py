import numpy as np
import matplotlib.pyplot as plt
import utility as u


def add_noise(pattern, percentage):
    noisy_pattern = pattern.copy()
    shuffle_size = int(percentage * len(pattern))
    positions = np.random.choice(pattern.shape[0], shuffle_size, replace=False)
    old_values = noisy_pattern[positions]
    new_values = np.where(old_values == 1, -1, 1)
    noisy_pattern[positions] = new_values
    return noisy_pattern


def plot_stats(patterns, x, y, xlabel, ylabel):
    for i, pattern in enumerate(patterns):
        plt.plot(x, y[i], label='p' + str(i))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.xlim([x[0], x[-1]])
    plt.show()


np.random.seed(100)

# load and store pictures
patterns = u.load_pictures()
stored_patterns = patterns[0:3]
hn = u.store_pictures(stored_patterns)

noise_levels = np.arange(0.1, 1, 0.1)
amount_recovered = np.zeros((len(stored_patterns), len(noise_levels)))
noise_removed = np.zeros((len(stored_patterns), len(noise_levels)))

# add noise and try to remove it
n_runs = 10
for n in range(n_runs):
    for i, pattern in enumerate(stored_patterns):
        for j, noise_percentage in enumerate(noise_levels):
            distort_pic = add_noise(pattern, noise_percentage)
            recall_pic = hn.recall(distort_pic, update_rule='synch')[0]

            # check recovery
            n_wrong_pixels = len(np.where(recall_pic != pattern)[0])
            n_noisy_pixels = noise_percentage * len(pattern)
            nr = (n_noisy_pixels - n_wrong_pixels) / n_noisy_pixels
            noise_removed[i, j] += nr
            if nr == 1:
                amount_recovered[i, j] += 1
            else:
                amount_recovered[i, j] += 0
noise_removed /= n_runs
amount_recovered /= n_runs

# plot removed noise
plot_stats(stored_patterns, noise_levels, noise_removed, '% noise', '% removed noise')
plot_stats(stored_patterns, noise_levels, amount_recovered, '% noise', '# times converged to right attractor')
