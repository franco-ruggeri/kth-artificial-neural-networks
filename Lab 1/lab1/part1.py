import numpy as np
import matplotlib.pyplot as plt
from perceptron import SLP, MLP
from mpl_toolkits import mplot3d


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


def plot_learning_curve(perceptron, axes, style, label, plot_val=False):

    if plot_val:
        axes.plot(perceptron.val_mce, style, label=label)

    else:
        axes.plot(perceptron.mce, style, label=label)
    # print(perceptron.val_mce)
    # print(perceptron.mce)


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


def linearly_non_separable_ml(patterns, targets,
                              axes_db, axes_lc, xlim, seed, learning_rate, epochs, hidden_nodes):
    np.random.seed(seed)

    # train
    targets = np.where(targets == 1, 1, -1)     # targets {-1,1} for delta rule
    perceptron = MLP(learning_rate, epochs, hidden_nodes)
    train_patterns, train_targets, val_patterns, val_targets = sub_sample_data(patterns, targets, seed)
    # print(train_targets)
    # print(val_targets)

    colors = ['magenta', 'green', 'black', 'yellow'] #, 'purple']
    labels = ['scenario 1', 'scenario 2', 'scenario 3', 'scenario 4']
    n_hidden_nodes = [1, 2, 3, 4, 5]

    # for i in range(len(n_hidden_nodes)):
    #     perceptron = MLP(learning_rate, epochs, n_hidden_nodes[i])
    #     perceptron.learn(patterns, targets, None, None)
    #     plot_learning_curve(perceptron, axes_lc, '-', label="LC for " + str(n_hidden_nodes[i]) + " hidden nodes")
    #     N = 1000
    #     x1 = np.linspace(xlim[0], xlim[1], N)
    #     x2 = np.linspace(xlim[2], xlim[3], N)
    #     xx1, xx2 = np.meshgrid(x1, x2)
    #     xx = np.concatenate((np.ravel(xx1).reshape(1, -1), np.ravel(xx2).reshape(1, -1)), axis=0)
    #     y = perceptron.predict(xx, classification=True).reshape(N, N)
    #     axes_db.contour(xx1, xx2, y, levels=(0,), colors=colors[i])
    #     proxy = plt.Rectangle((0, 0), 1, 1, fc=colors[i], label='Number of hidden nodes: ' + str(n_hidden_nodes[i]))
    #     axes_db.patches += [proxy]

    for i in range(len(train_patterns)):

        # print(train_patterns[i].shape)
        # print(train_targets[i].shape)
        # print(val_patterns[i].shape)
        # print(val_targets[i].shape)
        # print(train_patterns[i])
        perceptron.learn(train_patterns[i], train_targets[i], val_patterns[i], val_targets[i])

        # learning curve
        plot_learning_curve(perceptron, axes_lc, '-', label='train ' + labels[i], plot_val=False)
        plot_learning_curve(perceptron, axes_lc, '-', label='val' + labels[i], plot_val=True)

        # decision boundary
        N = 1000
        x1 = np.linspace(xlim[0], xlim[1], N)
        x2 = np.linspace(xlim[2], xlim[3], N)
        xx1, xx2 = np.meshgrid(x1, x2)
        xx = np.concatenate((np.ravel(xx1).reshape(1, -1), np.ravel(xx2).reshape(1, -1)), axis=0)
        y = perceptron.predict(xx, classification=True).reshape(N, N)
        axes_db.contour(xx1, xx2, y, levels=(0,), colors=colors[i])
        proxy = plt.Rectangle((0, 0), 1, 1, fc=colors[i], label=labels[i])
        axes_db.patches += [proxy]


def auto_encoder(learning_rate):

    data = -1*np.ones((8,8))
    np.fill_diagonal(data,1)
    perceptron = MLP(learning_rate, 100, hidden_nodes=2)
    epochs, array = perceptron.learn(data, data, None, None)
    print(epochs)
    print(array)

def gaussian(x, y):
    return np.exp(-(x**2 + y**2)/10) - 0.5


def function_approximation(learning_rate, epochs, hidden_nodes, nsamp, plot_decision_surface=False):

    x = np.arange(-5, 5, 0.5)
    y = np.arange(-5, 5, 0.5)
    ndata = len(x)*len(y)
    xx, yy = np.meshgrid(x, y)

    z = gaussian(xx, yy)
    ax = plt.axes(projection='3d')

    targets = np.reshape(z, (1, ndata))
    patterns = np.concatenate((np.reshape(xx, (1, ndata)), np.reshape(yy, (1, ndata))))

    # sub sample
    shuffled_patterns, shuffled_targets = shuffle_data(patterns, targets, 10)
    training_patterns, training_targets = shuffled_patterns[:, 0:nsamp], shuffled_targets[:, 0:nsamp]
    validation_patterns, validation_targets = shuffled_patterns[:, nsamp:], shuffled_targets[:, nsamp:]

    perceptron = MLP(learning_rate, epochs, hidden_nodes)
    perceptron.learn(training_patterns, training_targets, validation_patterns, validation_targets)

    zz = perceptron.predict(patterns, classification=False).reshape(20, 20)
    zz_train = perceptron.predict(training_patterns, classification=False)

    # train_mse = np.sum(np.square(zz_train - training_targets))/np.shape(training_targets)[1]
    # val_mse = np.sum(np.square(zz - targets))/np.shape(targets)[1]

    #plot_mse(perceptron)
    if plot_decision_surface:
        #ax.contour3D(xx, yy, z, 50, cmap='viridis')

        ax.contour3D(xx, yy, zz, 50, cmap='viridis')
        plt.show()

    #return train_mse, val_mse

def plot_mse(perceptron):

    """Plots how the MSE varies over epochs, probably not a necessary metric for this report though"""
    mse = perceptron.mse_over_time
    val_mse = perceptron.val_mse_over_time
    plt.figure()
    plt.title('Number of hidden nodes:' + str(perceptron.hidden_nodes))
    plt.plot(np.arange(len(mse)), mse, label='training MSE')
    plt.plot(np.arange(len(val_mse)), val_mse, label='validation MSE')
    plt.legend()
    plt.show()

seed = 5

# prepare figures
# fig_db = plt.figure()
# axes_db = plt.axes()
# fig_lc = plt.figure()
# axes_lc = plt.axes()

# generate data
mu = [(1.0, .3), (0, -.1)]
sigma = [.2, .3]
N = 100
patterns, targets = generate_data(mu, sigma, N, seed, linearly_sep=False)
patterns, targets = shuffle_data(patterns, targets, seed)
#train_patterns, train_targets, val_patterns, val_targets = sub_sample_data(patterns, targets, seed)
xlim = [min(patterns[0])-0.1, max(patterns[0])+0.1, min(patterns[1])-0.1, max(patterns[1])+0.1]
#plot_data(axes_db, patterns, targets)

# analysis
# perceptron_vs_delta(patterns, targets, axes_db, axes_lc, xlim, seed, learning_rates=[0.01, 0.001, 0.0001], epochs=30)
# batch_vs_sequential(patterns, targets, axes_db, axes_lc, xlim, seed,
#                     learning_rates=[0.1, 0.001], epochs=20, n_orders=1, method='delta')
# batch_vs_sequential(patterns, targets, axes_db, axes_lc, xlim, seed,
#                     learning_rates=[0.1], epochs=20, n_orders=2, method='perceptron')
# no_bias(patterns, targets, axes_db, axes_lc, xlim, seed, learning_rate=0.001, epochs=20)
# linearly_non_separable_sl(patterns, targets, axes_db, xlim, seed, learning_rate=0.001, epochs=100, n_runs=10)
#linearly_non_separable_ml(patterns, targets, axes_db, axes_lc, xlim, seed,
#                          learning_rate=0.1, epochs=200, hidden_nodes=5)

#auto_encoder(learning_rate=0.01)

# PART 3.2.3.1
# for i in range(1, 26):
#     print("Number of hidden nodes:", i)
#     mse.append(function_approximation(learning_rate=0.1, epochs=100, hidden_nodes=i, nsamp=100, plot_decision_surface=False))
# print(mse, len(mse))

function_approximation(learning_rate=0.5, epochs=100, hidden_nodes=25, nsamp=380, plot_decision_surface=True)

# PART 3.2.3.2
# mse = []
# val_mse = []
# nsamp_percentages = np.linspace(0.20, 0.80, 7)
# nsamp_distribution = 400*nsamp_percentages
# nsamp_distribution = nsamp_distribution.astype(int)
# for nsamp in nsamp_distribution:
#     train_mse, validation_mse = function_approximation(learning_rate=0.1, epochs=100, hidden_nodes=25, nsamp=nsamp)
#     mse.append(train_mse)
#     val_mse.append(validation_mse)
#
#
# plt.plot(nsamp_percentages, mse, label='training MSE')
# plt.plot(nsamp_percentages, val_mse, label='validation MSE')
# plt.xlabel('number of samples, 'r'[%]')
# plt.ylabel('Mean Squared Error')
# plt.legend()
# plt.show()


# show figures
# axes_db.legend(loc='lower right')
# axes_lc.legend(loc='upper right')
# axes_db.set_xlabel(r'$x_1$')
# axes_db.set_ylabel(r'$x_2$')
# axes_lc.set_xlabel('epoch')
# axes_lc.set_ylabel('misclassification error')
# axes_db.axis(xlim)
#fig_db.show()
#fig_lc.show()

# plt.show()
