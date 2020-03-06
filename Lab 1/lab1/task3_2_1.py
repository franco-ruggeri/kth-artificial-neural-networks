from utility import *
from perceptron import MLP

seed = 10
plt.close('all')

# generate data
patterns, targets = generate_linearly_non_separable_data(N=100, mu=[[1, .3], [0, -.1]], sigma=[.2, .3], seed=seed)
xlim = (min(patterns[0, :])-.2, max(patterns[0, :])+.2, min(patterns[1, :])-.2, max(patterns[1, :]+.2))

# train with all data
colors = ['y', 'm', 'k']
contours = []
labels = []
for i, n_hidden in enumerate(range(2, 5)):
    # train
    perceptron = MLP(learning_rate=0.1, n_epochs=10000, n_hidden=n_hidden, mode='batch', seed=seed)
    perceptron.learn(patterns, targets)

    # plot decision boundary
    plt.figure(1)
    c = plot_decision_boundary_mlp(perceptron, xlim, color=colors[i])
    contours.append(c)
    labels.append('n_hidden={}'.format(n_hidden))

    # plot decision boundary
    plt.figure(2)
    plot_learning_curve(perceptron, label=labels[-1], style=colors[i])

plt.figure(1)
plot_data(patterns, targets)
artists = [c.legend_elements()[0] for c in contours]
plt.legend([a[0] for a in artists], labels, loc='upper right')
plt.figure(2)
plt.legend()
plt.show()

# train with training set and test with validation set
colors = ['y', 'm', 'k', 'g']
contours = []
labels = []
for scenario, percentages in enumerate([(.75, .75, .75), (.5, .5, 1), (1, 1, .5), (.8, .2, 1)]):
    # subsample
    train_p, train_t, val_p, val_t = subsample_data(patterns, targets, percentages)

    # train
    perceptron = MLP(learning_rate=0.1, n_epochs=10000, n_hidden=3, mode='batch', seed=seed)
    perceptron.learn(train_p, train_t, val_p, val_t)

    # plot decision boundary
    # plt.figure(1)
    # c = plot_decision_boundary_mlp(perceptron, xlim, color=colors[scenario])
    # contours.append(c)
    # labels.append('scenario {}'.format(scenario+1))

    # plot learning curve
    # plt.figure(2)
    # plot_learning_curve(perceptron, label=labels[-1])

# plt.figure(1)
# plot_data(patterns, targets)
# artists = [c.legend_elements()[0] for c in contours]
# plt.legend([a[0] for a in artists], labels, loc='upper right')
# plt.figure(2)
# plt.legend()
# plt.show()
