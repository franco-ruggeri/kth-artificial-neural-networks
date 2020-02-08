import numpy as np
import matplotlib.pyplot as plt
from SOM import SOM


def load_animals(filename_attributes, filename_names):
    with open(filename_attributes) as f:
        attributes = f.read()
    attributes = np.array([int(a) for a in attributes.split(',')]).reshape(-1, 84)
    with open(filename_names) as f:
        names = f.readlines()
    names = [n.strip().strip("'").strip() for n in names]  # remove ', \t, \n
    return attributes, names


def sort_animals(som, attributes, names):
    idx = list(range(attributes.shape[0]))  # indexes of animals
    winners = [som.winner(a) for a in attributes]
    aux = zip(idx, winners)
    aux = sorted(aux, key=lambda x: x[1])  # sort by winner
    idx = [i[0] for i in aux]
    names = [names[i] for i in idx]
    return names  # animals in natural order


def load_cities(filename):
    with open(filename) as f:
        cities = f.readlines()
    cities = np.array([[float(x.strip().strip(';')) for x in c.split(',')]
                       for c in cities if c[0] != '%' and c[0] != '\n'])
    return cities


def plot_tour(som, cities):
    # we want a cyclic curve, so we add the first point at the end (the last point gets connected to the first one)
    weights = np.concatenate((som.weights, som.weights[0].reshape(1, -1)), axis=0)
    plt.scatter(cities[:, 0], cities[:, 1], c='y', label='city')
    plt.plot(weights[:, 0], weights[:, 1], 'b-', label='tour')
    plt.plot(weights[:, 0], weights[:, 1], 'ro', label='tour')
    plt.legend()
    plt.show()


def compute_tour_length(som):
    tour_length = 0
    for i in range(som.n_nodes):
        x = som.weights[i]
        y = som.weights[(i + 1) % som.n_nodes]
        tour_length += np.linalg.norm(x - y)
    return tour_length


def load_mp_info(filename):
    with open(filename) as f:
        values = f.readlines()
    values = [int(v.strip()) for v in values if v[0] != '%' and v.strip()]
    return values


def load_mp(filename_votes, filename_parties, filename_genders, filename_districts):
    with open(filename_votes) as f:
        votes = f.read()
    votes = np.array([float(p) for p in votes.split(',')]).reshape(-1, 31)
    parties = load_mp_info(filename_parties)
    genders = load_mp_info(filename_genders)
    districts = load_mp_info(filename_districts)
    return votes, parties, genders, districts


def plot_mp_colored(winners, colors, side, title, labels=None):
    # colors
    winner_colors = [[[] for j in range(side)] for i in range(side)]
    for i in range(len(colors)):
        winner = winners[i]
        winner_colors[winner[0]][winner[1]].append(colors[i])
    for i in range(side):
        for j in range(side):
            if len(winner_colors[i][j]) != 0:
                winner_colors[i][j] = sum(winner_colors[i][j]) / len(winner_colors[i][j])
            else:
                winner_colors[i][j] = 0
    plt.imshow(winner_colors)
    plt.axis('off')
    plt.title(title)

    # texts
    if labels is not None:
        colors = [c-min(colors) for c in colors]    # start from 0 to index correctly the labels
        winner_labels = [[[0 for k in range(len(labels))] for j in range(side)] for i in range(side)]
        for i in range(len(colors)):    # count
            winner = winners[i]
            winner_labels[winner[0]][winner[1]][colors[i]] += 1
        for i in range(side):
            for j in range(side):
                label = ''
                for k in range(len(labels)):
                    if winner_labels[i][j][k] > 0:
                        label += str(winner_labels[i][j][k]) + ' ' + labels[k] + '\n'
                plt.text(j, i, label, fontsize=5, ha='center', va='center')
    plt.show()


def plot_mp(som, votes, parties, genders, districts):
    side = som.weights.shape[0]
    winners = [som.winner(v) for v in votes]

    # parties
    labels = ['no party', 'm', 'fp', 's', 'v', 'mp', 'kd', 'c']
    plot_mp_colored(winners, parties, side, 'Output space (colored by party)', labels=labels)

    # genders
    labels = ['male', 'female']
    genders = [g+1 for g in genders]    # to distinguish the color between nobody and men in the grid
    plot_mp_colored(winners, genders, side, 'Output space (colored by gender)', labels=labels)

    # districts
    # labels = [str(d) for d in districts]
    # plot_mp_colored(winners, districts, side, labels=labels)
    plot_mp_colored(winners, districts, side, 'Output space (colored by district)')


np.random.seed(1)

# 4.1 - Animals
attributes, names = load_animals('datasets/animals.dat', 'datasets/animalnames.txt')
som = SOM(dim=1, n_nodes=100, learning_rate=0.2, n_epochs=20, init_nb_size=50)
som.learn(attributes)
names = sort_animals(som, attributes, names)
print('Sorted animals:', names)

# 4.2 - Cyclic tour
cities = load_cities('datasets/cities.dat')
som = SOM(dim=1, n_nodes=10, learning_rate=0.5, n_epochs=20, init_nb_size=2, circular=True)
som.learn(cities)
# plot_tour(som, cities)
tour_length = compute_tour_length(som)
print('Tour length:', tour_length)

# 4.3 - Votes of MPs
votes, parties, genders, districts = load_mp('datasets/votes.dat', 'datasets/mpparty.dat',
                                             'datasets/mpsex.dat', 'datasets/mpdistrict.dat')
som = SOM(dim=2, n_nodes=100, learning_rate=0.5, n_epochs=50, init_nb_size=10)
som.learn(votes)
plot_mp(som, votes, parties, genders, districts)
