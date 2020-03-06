import numpy as np
import matplotlib.pyplot as plt
from perceptron import MLP


def generate_linearly_non_separable_data(N, mu, sigma, seed=None):
    """Generate linearly non-separable data."""
    if seed is not None:
        np.random.seed(seed)
    patterns = np.zeros((2, 2*N))
    patterns[0, :N] = np.concatenate((
        np.random.randn(1, round(N / 2)) * sigma[0] - mu[0][0],
        np.random.randn(1, round(N / 2)) * sigma[0] + mu[0][0]
    ), axis=1)
    patterns[1, :N] = np.random.randn(1, N) * sigma[0] + mu[0][1]
    patterns[0, N:] = np.random.randn(1, N) * sigma[1] + mu[1][0]
    patterns[1, N:] = np.random.randn(1, N) * sigma[1] + mu[1][1]
    targets = np.concatenate((np.ones(N), -np.ones(N))).reshape(1, -1)
    return patterns, targets


def shuffle_data(patterns, targets, seed=None):
    if seed is not None:
        np.random.seed(seed)
    N = patterns.shape[1]
    idx = np.arange(N)
    np.random.shuffle(idx)
    return patterns[:, idx], targets[:, idx]


def subsample_data(patterns, targets, percentages, seed=None):
    """
    Subsample data.

    Args:
        patterns: patterns, shape (n_features, n_samples)
        targets: targets, shape (n_labels, n_samples)
        percentages: percentages of data for the training set
            position 0 -> class A left
            position 1 -> class A right
            position 2 -> class B
        seed: seed for random numbers
    """
    if seed is not None:
        np.random.seed(seed)

    labels = np.unique(targets)
    idx_a_left = np.where((targets[0] == labels[1]) & (patterns[0] < 0))[0]
    idx_a_right = np.where((targets[0] == labels[1]) & (patterns[0] > 0))[0]
    idx_b = np.where(targets == labels[0])[1]

    np.random.shuffle(idx_a_left)
    np.random.shuffle(idx_a_right)
    np.random.shuffle(idx_b)

    N_a_left = len(idx_a_left)
    N_a_right = len(idx_a_right)
    N_b = len(idx_b)

    p_a_left = percentages[0]
    p_a_right = percentages[1]
    p_b = percentages[2]

    idx_train = np.concatenate((
        idx_a_left[:round(p_a_left * N_a_left)],
        idx_a_right[:round(p_a_right * N_a_right)],
        idx_b[:round(p_b * N_b)]
    ))
    idx_test = np.concatenate((
        idx_a_left[round(p_a_left * N_a_left):],
        idx_a_right[round(p_a_right * N_a_right):],
        idx_b[round(p_b * N_b):]
    ))

    training_patterns = patterns[:, idx_train]
    training_targets = targets[:, idx_train]
    test_patterns = patterns[:, idx_test]
    test_targets = targets[:, idx_test]

    return training_patterns, training_targets, test_patterns, test_targets


def plot_data(patterns, targets):
    labels = np.unique(targets)
    idx = np.where(targets == labels[1])[1]
    plt.scatter(patterns[0, idx], patterns[1, idx], c='r', marker='*', label='class A')
    idx = np.where(targets == labels[0])[1]
    plt.scatter(patterns[0, idx], patterns[1, idx], c='b', marker='+', label='class B')
    xlim = (min(patterns[0, :]) - .2, max(patterns[0, :]) + .2, min(patterns[1, :]) - .2, max(patterns[1, :] + .2))
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(xlim[2], xlim[3])


def get_decision_boundary_slp(weights, bias, xlim):
    if not bias:
        weights = np.concatenate((weights, np.zeros(1)))
    x1 = np.linspace(xlim[0], xlim[1], 2)
    x2 = (-weights[2] - weights[0] * x1) / weights[1]
    return x1, x2


def plot_decision_boundary_slp(perceptron, xlim, style='', label=''):
    x1, x2 = get_decision_boundary_slp(perceptron.weights[0], perceptron.bias, xlim)
    plt.plot(x1, x2, style, label=label)


def plot_decision_boundary_mlp(perceptron, xlim, color=None):
    N = 1000
    x1 = np.linspace(xlim[0], xlim[1], N)
    x2 = np.linspace(xlim[2], xlim[3], N)
    xx1, xx2 = np.meshgrid(x1, x2)
    xx = np.concatenate((np.ravel(xx1).reshape(1, -1), np.ravel(xx2).reshape(1, -1)), axis=0)
    y = perceptron.predict(xx, thresholded=False).reshape(N, N)
    if color is None:
        c = plt.contour(xx1, xx2, y, levels=(0,))
    else:
        c = plt.contour(xx1, xx2, y, levels=(0,), colors=color)
    return c


def plot_learning_curve(perceptron, style='', label=''):
    if isinstance(perceptron, MLP) and perceptron.mce_val is not None:
        plt.plot(perceptron.mce_val, style, label=label+' validation')
        label += ' training'
    plt.plot(perceptron.mce, style, label=label)
    plt.xlabel('epoch')
    plt.ylabel('misclassification error')
    if perceptron.n_epochs > 1000:
        step = 1000
    elif perceptron.n_epochs > 100:
        step = 20
    else:
        step = 5
    plt.xticks(np.arange(0, perceptron.n_epochs+1, step))
