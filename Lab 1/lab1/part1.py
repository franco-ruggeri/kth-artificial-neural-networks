import numpy as np
import matplotlib.pyplot as plt
from perceptron import SLP, MLP


def generate_data(mu, sigma, N, seed, linearly_sep):
    np.random.seed(seed)
    patterns = np.zeros((2, 2 * N))
    if linearly_sep:
        patterns[0, :N] = np.random.randn(1, N) * sigma[0] + mu[0][0]
    else:
        patterns[0, :N] = np.concatenate((
            np.random.randn(1, round(N / 2)) * sigma[0] - mu[0][0],
            np.random.randn(1, round(N / 2)) * sigma[0] + mu[0][0]
        ), axis=1)
    patterns[1, :N] = np.random.randn(1, N) * sigma[0] + mu[0][1]
    patterns[0, N:] = np.random.randn(1, N) * sigma[1] + mu[1][0]
    patterns[1, N:] = np.random.randn(1, N) * sigma[1] + mu[1][1]
    targets = np.concatenate((np.ones(N), np.zeros(N))).reshape(1, -1)
    return patterns, targets


def shuffle_data(patterns, targets, seed):
    np.random.seed(seed)
    N = patterns.shape[1]
    idx = np.arange(N)
    np.random.shuffle(idx)
    return patterns[:, idx], targets[:, idx]


def sub_sample_data(patterns, targets, seed):
    labels = np.unique(targets)
    training_patterns = []
    training_targets = []
    test_patterns = []
    test_targets = []

    # sub-samples 1-3
    for percentageA, percentageB in [(.75, .75), (.5, 1.), (1., .5)]:
        np.random.seed(seed)
        idx_a = np.where(targets == labels[1])[1]
        idx_b = np.where(targets == labels[0])[1]
        np.random.shuffle(idx_a)
        np.random.shuffle(idx_b)
        idx_train = np.concatenate((idx_a[:round(percentageA * len(idx_a))], idx_b[:round(percentageB * len(idx_b))]))
        idx_test = np.concatenate((idx_a[round(percentageA * len(idx_a)):], idx_b[round(percentageB * len(idx_b)):]))
        training_patterns.append(patterns[:, idx_train])
        training_targets.append(targets[:, idx_train])
        test_patterns.append(patterns[:, idx_test])
        test_targets.append(targets[:, idx_test])

    # sub-sample 4
    np.random.seed(seed)
    idx_a_left = np.where((targets[0] == labels[1]) & (patterns[0] < 0))[0]
    idx_a_right = np.where((targets[0] == labels[1]) & (patterns[0] > 0))[0]
    np.random.shuffle(idx_a_left)
    np.random.shuffle(idx_a_right)
    idx_train = np.concatenate((
        idx_a_left[:round(.8 * len(idx_a_left))],
        idx_a_right[:round(.2 * len(idx_a_right))],
        np.where(targets == labels[0])[1]
    ))
    idx_test = np.concatenate((
        idx_a_left[round(.8 * len(idx_a_left)):],
        idx_a_right[round(.2 * len(idx_a_right)):],
    ))
    training_patterns.append(patterns[:, idx_train])
    training_targets.append(targets[:, idx_train])
    test_patterns.append(patterns[:, idx_test])
    test_targets.append(targets[:, idx_test])

    return training_patterns, training_targets, test_patterns, test_targets


def plot_data(plot, patterns, targets):
    labels = np.unique(targets)
    idx = np.where(targets == labels[1])[1]
    plot.scatter(patterns[0, idx], patterns[1, idx], c='r', marker='*', label='class A')
    idx = np.where(targets == labels[0])[1]
    plot.scatter(patterns[0, idx], patterns[1, idx], c='b', marker='.', label='class B')


def train_perceptron(patterns, targets, learning_rate, epochs, method, mode, seed, bias=True):
    np.random.seed(seed)

    # adjust labels
    if method == 'perceptron':
        targets = np.where(targets == 1, 1, 0)      # targets {-1,1} for delta rule
    elif method == 'delta':
        targets = np.where(targets == 1, 1, -1)     # targets {-1,1} for delta rule
    else:
        exit(-1)

    # train
    perceptron = SLP(learning_rate, epochs, method, mode, bias)
    perceptron.learn(patterns, targets)
    return perceptron


def evaluate_accuracy_perceptron(perceptron, patterns, targets):
    # adjust labels
    if perceptron.delta:
        targets = np.where(targets == 1, 1, -1)     # targets {-1,1} for delta rule
        labels = [1, -1]
    else:
        targets = np.where(targets == 1, 1, 0)      # targets {-1,1} for delta rule
        labels = [1, 0]

    # evaluate accuracy
    idx_a = np.where(targets == labels[0])[1]
    idx_b = np.where(targets == labels[1])[1]
    y = perceptron.predict(patterns)

    if len(idx_a) > 0:
        accuracy_a = len(np.where(targets[:, idx_a] == y[:, idx_a])[1]) / len(idx_a)
    else:
        accuracy_a = np.NaN
    if len(idx_b) > 0:
        accuracy_b = len(np.where(targets[:, idx_b] == y[:, idx_b])[1]) / len(idx_b)
    else:
        accuracy_b = np.NaN
    return accuracy_a, accuracy_b


def plot_decision_boundary(perceptron, axes, xlim, label, N=100):
    x1 = np.linspace(xlim[0], xlim[1], N)
    x2 = np.linspace(xlim[2], xlim[3], N)
    w = perceptron.weights[0, :]
    if not perceptron.bias:
        w = np.concatenate((w, np.zeros(1)))
    x2 = (-w[2] - w[0] * x1) / w[1]
    axes.plot(x1, x2, label=label, linewidth=2)


def plot_learning_curve(perceptron, axes, style, label):
    axes.plot(perceptron.mce, style, label=label)


def perceptron_vs_delta(patterns, targets, axes_db, axes_lc, xlim, seed, learning_rates, epochs):
    for learning_rate in learning_rates:
        # train
        p_perceptron = train_perceptron(patterns, targets, learning_rate, epochs, 'perceptron', 'batch', seed)
        p_delta = train_perceptron(patterns, targets, learning_rate, epochs, 'delta', 'batch', seed)

        # plot
        plot_decision_boundary(p_perceptron, axes_db, xlim, r'perceptron $\eta={}$'.format(learning_rate))
        plot_decision_boundary(p_delta, axes_db, xlim, r'delta $\eta={}$'.format(learning_rate))
        plot_learning_curve(p_perceptron, axes_lc, 'o-', r'perceptron $\eta={}$'.format(learning_rate))
        plot_learning_curve(p_delta, axes_lc, 'o-', r'delta $\eta={}$'.format(learning_rate))
    axes_db.set_title('Perceptron vs Delta - batch mode')
    axes_lc.set_title('Perceptron vs Delta - batch mode')
    axes_lc.set_xticks(np.arange(0, epochs + 1, 2))


def batch_vs_sequential(patterns, targets, axes_db, axes_lc, xlim, seed, learning_rates, epochs, n_orders, method):
    # generate different orders
    data = []
    for i in range(n_orders):
        data.append(shuffle_data(patterns, targets, seed=i*100+1))

    for learning_rate in learning_rates:
        for i in range(len(data)):
            # train
            p_batch = train_perceptron(data[i][0], data[i][1], learning_rate, epochs, method, 'batch', seed)
            p_seq = train_perceptron(data[i][0], data[i][1], learning_rate, epochs, method, 'sequential', seed)

            # plot
            label_end = r'$\eta={}$'.format(learning_rate)
            if len(data) > 1:
                label_end = str(i + 1) + ' ' + label_end
            plot_decision_boundary(p_batch, axes_db, xlim, 'batch mode ' + label_end)
            plot_decision_boundary(p_seq, axes_db, xlim, 'sequential mode ' + label_end)
            plot_learning_curve(p_batch, axes_lc, 'o-', 'batch mode ' + label_end)
            plot_learning_curve(p_seq, axes_lc, 'o-', 'sequential mode ' + label_end)
    axes_db.set_title('Batch vs sequential - {}'.format(method))
    axes_lc.set_title('Batch vs sequential - {}'.format(method))
    axes_lc.set_xticks(np.arange(0, epochs + 1, 2))


def no_bias(patterns, targets, axes_db, axes_lc, xlim, seed, learning_rate, epochs):
    # train
    p_no_bias = train_perceptron(patterns, targets, learning_rate, epochs, 'delta', 'batch', seed, bias=False)
    p_bias = train_perceptron(patterns, targets, learning_rate, epochs, 'delta', 'batch', seed, bias=True)

    # plot
    plot_decision_boundary(p_no_bias, axes_db, xlim, 'w/o bias weight')
    plot_decision_boundary(p_bias, axes_db, xlim, 'with bias weight')
    plot_learning_curve(p_no_bias, axes_lc, 'o-', 'w/o bias weight')
    plot_learning_curve(p_bias, axes_lc, 'o-', 'with bias weight')
    axes_db.set_title('Bias weight - Delta')
    axes_lc.set_title('Bias weight - Delta')
    axes_lc.set_xticks(np.arange(0, epochs + 1, 2))


def linearly_non_separable_sl(patterns, targets, axes_db, xlim, seed, learning_rate, epochs, n_runs):
    # decision boundary
    training_patterns, training_targets = sub_sample_data(patterns, targets, seed)[0:2]
    for i in range(len(training_patterns)):
        # train
        p_perceptron = train_perceptron(training_patterns[i], training_targets[i], learning_rate, epochs,
                                        'perceptron', 'batch', seed)
        p_delta = train_perceptron(training_patterns[i], training_targets[i], learning_rate, epochs,
                                   'delta', 'batch', seed)

        # plot
        plot_decision_boundary(p_perceptron, axes_db, xlim, 'perceptron, subsample {}'.format(i))
        plot_decision_boundary(p_delta, axes_db, xlim, 'delta, subsample {}'.format(i))
    axes_db.set_title('Perceptron vs Delta - subsample')

    # performance
    accuracy_perceptron_test = np.zeros((4, 2))
    accuracy_delta_test = np.zeros((4, 2))
    accuracy_perceptron = np.zeros((4, 2))
    accuracy_delta = np.zeros((4, 2))
    for n in range(n_runs):
        training_patterns, training_targets, test_patterns, test_targets = sub_sample_data(patterns, targets, n*100+1)
        for i in range(len(training_patterns)):
            # train
            p_perceptron = train_perceptron(training_patterns[i], training_targets[i], learning_rate, epochs,
                                            'perceptron', 'batch', seed, bias=False)
            p_delta = train_perceptron(training_patterns[i], training_targets[i], learning_rate, epochs,
                                       'delta', 'batch', seed, bias=False)

            # evaluate
            accuracy_perceptron[i] += evaluate_accuracy_perceptron(p_perceptron, patterns, targets)
            accuracy_delta[i] += evaluate_accuracy_perceptron(p_delta, patterns, targets)
            accuracy_perceptron_test[i] += evaluate_accuracy_perceptron(p_perceptron, test_patterns[i], test_targets[i])
            accuracy_delta_test[i] += evaluate_accuracy_perceptron(p_delta, test_patterns[i], test_targets[i])
    for i in range(accuracy_perceptron_test.shape[0]):
        print('Subsample', i)
        print('Accuracy perceptron class A:', accuracy_perceptron[i][0] / n_runs)
        print('Accuracy perceptron class B:', accuracy_perceptron[i][1] / n_runs)
        print('Accuracy perceptron class A test:', accuracy_perceptron_test[i][0] / n_runs)
        print('Accuracy perceptron class B test:', accuracy_perceptron_test[i][1] / n_runs)
        print('Accuracy delta class A:', accuracy_delta[i][0] / n_runs)
        print('Accuracy delta class B:', accuracy_delta[i][1] / n_runs)
        print('Accuracy delta class A test:', accuracy_delta_test[i][0] / n_runs)
        print('Accuracy delta class B test:', accuracy_delta_test[i][1] / n_runs)
        print()


def linearly_non_separable_ml(patterns, targets, axes_db, axes_lc, xlim, seed, learning_rate, epochs, hidden_nodes):
    np.random.seed(seed)

    # train
    targets = np.where(targets == 1, 1, -1)     # targets {-1,1} for delta rule
    perceptron = MLP(learning_rate, epochs, hidden_nodes)
    perceptron.learn(patterns, targets)

    # learning curve
    plot_learning_curve(perceptron, axes_lc, '-', None)

    # decision boundary
    N = 1000
    x1 = np.linspace(xlim[0], xlim[1], N)
    x2 = np.linspace(xlim[2], xlim[3], N)
    xx1, xx2 = np.meshgrid(x1, x2)
    xx = np.concatenate((np.ravel(xx1).reshape(1, -1), np.ravel(xx2).reshape(1, -1)), axis=0)
    y = perceptron.predict(xx).reshape(N, N)
    axes_db.contour(xx1, xx2, y, levels=(0,))


seed = 100

# prepare figures
fig_db = plt.figure()
axes_db = plt.axes()
fig_lc = plt.figure()
axes_lc = plt.axes()

# generate data
mu = [(2., .3), (0, -.1)]
sigma = [.4, .4]
N = 100
patterns, targets = generate_data(mu, sigma, N, seed, linearly_sep=True)
patterns, targets = shuffle_data(patterns, targets, seed)
xlim = [min(patterns[0])-0.1, max(patterns[0])+0.1, min(patterns[1])-0.1, max(patterns[1])+0.1]
plot_data(axes_db, patterns, targets)

# analysis
# perceptron_vs_delta(patterns, targets, axes_db, axes_lc, xlim, seed, learning_rates=[0.01, 0.001, 0.0001], epochs=30)
batch_vs_sequential(patterns, targets, axes_db, axes_lc, xlim, seed,
                    learning_rates=[0.1, 0.001], epochs=20, n_orders=1, method='delta')
# batch_vs_sequential(patterns, targets, axes_db, axes_lc, xlim, seed,
#                     learning_rates=[0.1], epochs=20, n_orders=2, method='perceptron')
# no_bias(patterns, targets, axes_db, axes_lc, xlim, seed, learning_rate=0.001, epochs=20)
# linearly_non_separable_sl(patterns, targets, axes_db, xlim, seed, learning_rate=0.001, epochs=100, n_runs=10)
# linearly_non_separable_ml(patterns, targets, axes_db, axes_lc, xlim, seed,
#                           learning_rate=0.1, epochs=200, hidden_nodes=3)

# show figures
axes_db.legend(loc='lower right')
axes_lc.legend(loc='upper right')
axes_db.set_xlabel(r'$x_1$')
axes_db.set_ylabel(r'$x_2$')
axes_lc.set_xlabel('epoch')
axes_lc.set_ylabel('misclassification error')
axes_db.axis(xlim)
fig_db.show()
fig_lc.show()
