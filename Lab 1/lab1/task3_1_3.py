from utility import *
from perceptron import SLP


def score(perceptron, test_pattern, test_targets):
    # adjust labels
    if perceptron.delta:
        test_targets = np.where(test_targets == 1, 1, -1)
        labels = [-1, 1]
    else:
        test_targets = np.where(test_targets == 1, 1, 0)
        labels = [0, 1]

    idx_a = np.where(test_targets == labels[1])[1]
    idx_b = np.where(test_targets == labels[0])[1]
    if len(idx_a) > 0:
        score_a = perceptron.score(test_pattern[:, idx_a], test_targets[:, idx_a])
    else:
        score_a = None
    if len(idx_b) > 0:
        score_b = perceptron.score(test_pattern[:, idx_b], test_targets[:, idx_b])
    else:
        score_b = None

    return score_a, score_b


seed = 1

# generate data
patterns, targets = generate_linearly_non_separable_data(N=100, mu=[[1, .3], [0, -.1]], sigma=[.2, .3], seed=seed)
xlim = (min(patterns[0, :])-.2, max(patterns[0, :])+.2, min(patterns[1, :])-.2, max(patterns[1, :]+.2))

N = 100     # number of runs to average results
for scenario, percentages in enumerate([(.75, .75, .75), (.5, .5, 1), (1, 1, .5), (.8, .2, 1)]):
    score_a = []
    score_b = []
    score_a_test = []
    score_b_test = []

    for n in range(N):
        # subsample
        train_p, train_t, test_p, test_t = subsample_data(patterns, targets, percentages, seed=seed+n)

        # train
        perceptron = SLP(learning_rate=0.001, n_epochs=500, rule='delta', mode='batch', seed=seed + n)
        perceptron.learn(train_p, train_t)

        # accuracy on original dataset
        score_a_, score_b_ = score(perceptron, patterns, targets)
        score_a.append(score_a_)
        score_b.append(score_b_)

        # accuracy on test set (generalisation)
        score_a_, score_b_ = score(perceptron, test_p, test_t)
        if score_a_ is not None:
            score_a_test.append(score_a_)
        if score_b_ is not None:
            score_b_test.append(score_b_)

        # plot (just for one run)
        if n == 0:
            plot_decision_boundary_slp(perceptron, xlim, label='scenario {}'.format(scenario + 1))

    print('Scenario', scenario+1)
    print('Score class A: {:.2f}'.format(sum(score_a) / len(score_a)))
    print('Score class B: {:.2f}'.format(sum(score_b) / len(score_b)))
    if len(score_a_test) > 0:
        print('Score class A test: {:.2f}'.format(sum(score_a_test) / len(score_a_test)))
    else:
        print('Score class A test: -')
    if len(score_b_test) > 0:
        print('Score class B test: {:.2f}'.format(sum(score_b_test) / len(score_b_test)))
    else:
        print('Score class B test: -')
    print()

plot_data(patterns, targets)
plt.legend()
plt.show()
