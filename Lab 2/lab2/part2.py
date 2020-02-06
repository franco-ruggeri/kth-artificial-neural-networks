import numpy as np
import matplotlib.pyplot as plt
from SOM import SOM


def load_animals(filename_attributes, filename_names):
    with open(filename_attributes) as f:
        patterns = f.read()
    patterns = np.array([int(p) for p in patterns.split(',')]).reshape(-1, 84)
    with open(filename_names) as f:
        names = f.readlines()
    names = [n.replace("'", "").replace('\t', '').replace('\n', '') for n in names]     # remove ', \t, \n
    return patterns, names


def sort_animals(som, patterns, names):
    idx = list(range(patterns.shape[0]))            # indexes of animals
    winners = [som.winner(p) for p in patterns]
    aux = zip(idx, winners)
    aux = sorted(aux, key=lambda x: x[1])           # sort by winner
    idx = [i[0] for i in aux]
    names = [names[i] for i in idx]
    return names                                    # animals in natural order


def load_cities(filename):
    with open(filename) as f:
        patterns = f.readlines()
    patterns = np.array([[float(x.replace(' ', '').replace(';', '').replace('\n', '')) for x in p.split(',')]
                         for p in patterns if p[0] != '%' and p[0] != '\n'])
    return patterns


def plot_tour(som, patterns):
    plt.scatter(patterns[:, 0], patterns[:, 1], c='y', label='city')
    # we want a cyclic curve, so we add the first point at the end (the last point gets connected to the first one)
    weights = np.concatenate((som.weights, som.weights[0].reshape(1, -1)), axis=0)
    plt.plot(weights[:, 0], weights[:, 1], 'b-', label='tour')
    plt.plot(weights[:, 0], weights[:, 1], 'ro', label='tour')
    plt.legend()
    plt.show()


def compute_tour_length(som):
    tour_length = 0
    for i in range(som.n_nodes):
        x = som.weights[i]
        y = som.weights[(i+1) % som.n_nodes]
        tour_length += np.linalg.norm(x-y)
    return tour_length


np.random.seed(1)

# 4.1 - Animals
patterns, names = load_animals('datasets/animals.dat', 'datasets/animalnames.txt')
som = SOM(n_nodes=100, learning_rate=0.2, n_epochs=20, init_nb_size=50, verbose=False)
som.learn(patterns)
names = sort_animals(som, patterns, names)
print('Sorted animals:', names)

# 4.2 - Cyclic tour
patterns = load_cities('datasets/cities.dat')
som = SOM(n_nodes=10, learning_rate=0.5, n_epochs=20, init_nb_size=2, circular=True, verbose=False)
som.learn(patterns)
plot_tour(som, patterns)
tour_length = compute_tour_length(som)
print('Tour length:', tour_length)

