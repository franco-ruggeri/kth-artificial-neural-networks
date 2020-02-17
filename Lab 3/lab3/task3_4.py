import numpy as np
import matplotlib.pyplot as plt
import utility as u


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

step_size = 0.01
noise_levels = np.arange(0.1, 1+step_size, step_size)
amount_recovered = np.zeros((len(stored_patterns), len(noise_levels)))
noise_removed = np.zeros((len(stored_patterns), len(noise_levels)))

# add noise and try to remove it
n_runs = 100
for n in range(n_runs):
    for i, pattern in enumerate(stored_patterns):
        for j, noise_percentage in enumerate(noise_levels):
            distort_pic = u.add_noise(pattern, noise_percentage)
            recall_pic = hn.recall(distort_pic, synchronous=True)[0]

            # check recovery
            n_wrong_pixels = len(np.where(recall_pic != pattern)[0])
            n_noisy_pixels = noise_percentage * len(pattern)
            nr = (n_noisy_pixels - n_wrong_pixels) / n_noisy_pixels
            noise_removed[i, j] += nr
            if nr == 1:
                amount_recovered[i, j] += 1
            else:
                amount_recovered[i, j] += 0

            # set n_runs=1, step_size=0.1 and uncomment this to plot the attractors
            # u.plot_picture(recall_pic, title='{:d}% noise'.format(int(noise_percentage*100)))
noise_removed /= n_runs
amount_recovered /= n_runs

# plot removed noise
plot_stats(stored_patterns, noise_levels, noise_removed, '% noise', '% removed noise')
plot_stats(stored_patterns, noise_levels, amount_recovered, '% noise', '% convergence to right attractor')
