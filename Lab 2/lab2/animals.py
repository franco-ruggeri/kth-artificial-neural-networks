import numpy as np
from SOM import SOM


def load_animals(filename_attributes, filename_names):
    with open(filename_attributes) as f:
        patterns = f.read()
        patterns = np.array([int(p) for p in patterns.split(',')]).reshape(-1, 84)
    with open(filename_names) as f:
        names = f.readlines()
        names = [n.replace("'", "").replace('\t', '').replace('\n', '') for n in names]     # remove ', \t, \n
    return patterns, names


def get_natural_order(som, patterns):
    idx = list(range(patterns.shape[0]))            # indexes of animals
    winners = [som.winner(p) for p in patterns]
    aux = zip(idx, winners)
    aux = sorted(aux, key=lambda x: x[1])           # sort by winner
    idx = [i[0] for i in aux]
    return idx                                      # indexes of animals in natural order


patterns, names = load_animals('datasets/animals.dat', 'datasets/animalnames.txt')
som = SOM(n_nodes=100, learning_rate=0.2, n_epochs=20)
som.learn(patterns)
idx = get_natural_order(som, patterns)
for i in idx:
    print(names[i])
