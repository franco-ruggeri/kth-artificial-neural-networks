import numpy as np
import matplotlib.pyplot as plt
from perceptron import SingleLayerPerceptron, TwoLayerPerceptron


def plot_data(plot, patterns, targets):
    labels = np.unique(targets)
    idx = np.where(targets == labels[1])[1]
    plot.scatter(patterns[0, idx], patterns[1, idx], c='r', marker='*')#, label='class A')
    idx = np.where(targets == labels[0])[1]
    plot.scatter(patterns[0, idx], patterns[1, idx], c='b', marker='.')#, label='class B')


###################
# Data generation #
###################

# generate data
mu = [(1., .3), (0., -.1)]
sigma = [.2, .3]
N = 100
np.random.seed(1)
patterns = np.zeros((2, 2*N))
# patterns[0, :N] = np.random.randn(1, N) * sigma[0] + mu[0][0]
patterns[0, :N] = np.concatenate((
    np.random.randn(1, round(N/2)) * sigma[0] - mu[0][0],
    np.random.randn(1, round(N/2)) * sigma[0] + mu[0][0]
), axis=1)
patterns[1, :N] = np.random.randn(1, N) * sigma[0] + mu[0][1]
patterns[0, N:] = np.random.randn(1, N) * sigma[1] + mu[1][0]
patterns[1, N:] = np.random.randn(1, N) * sigma[1] + mu[1][1]
targets = np.concatenate((np.ones(N), np.zeros(N)))
xlim = [min(patterns[0])-0.1, max(patterns[0])+0.1, min(patterns[1])-0.1, max(patterns[1])+0.1]

# shuffle generated data
np.random.seed(1)
idx = np.arange(2*N)
np.random.shuffle(idx)
patterns = patterns[:, idx]
targets = targets[idx].reshape(1, -1)

# prepare figures
fig_db = plt.figure()
plot_db = plt.axes()
fig_lc = plt.figure()
plot_lc = plt.axes()

# plot generated data
plot_data(plot_db, patterns, targets)


#####################################
# Perceptron learning vs Delta rule #
#####################################

epochs = 30
for learning_rate in [0.001]:
    # train
    targets = np.where(targets == 1, 1, 0)  # targets {0,1} for perceptron learning
    np.random.seed(1)
    p_perceptron = SingleLayerPerceptron(learning_rate=learning_rate, epochs=epochs, method='perceptron',
                                         mode='batch', animation=False)
    p_perceptron.learn(patterns, targets)
    targets = np.where(targets == 1, 1, -1)  # targets {-1,1} for delta rule
    np.random.seed(1)
    p_delta = SingleLayerPerceptron(learning_rate=learning_rate, epochs=epochs, method='delta', mode='batch')
    p_delta.learn(patterns, targets)

    # plot decision boundaries and learning curves
#     p_perceptron.plot_decision_boundary(plot_db, xlim, r'perceptron $\eta={}$'.format(learning_rate))
#     p_delta.plot_decision_boundary(plot_db, xlim, r'delta $\eta={}$'.format(learning_rate))
#     p_perceptron.plot_learning_curve(plot_lc, r'perceptron $\eta={}$'.format(learning_rate))
#     p_delta.plot_learning_curve(plot_lc, r'delta $\eta={}$'.format(learning_rate))
# plot_db.set_title('Perceptron vs Delta - batch mode')
# plot_lc.set_title('Perceptron vs Delta - batch mode')
# plot_lc.set_xticks(np.arange(0, epochs + 1, 2))


#######################
# Batch vs sequential #
#######################

# generate other order
np.random.seed(100)
idx = np.arange(2*N)
np.random.shuffle(idx)
patterns_2 = patterns[:, idx]
targets_2 = targets[:, idx]

epochs = 30
for learning_rate in [0.1, 0.001]:
    i = 1
    for p, t in [(patterns, targets), (patterns_2, targets_2)]:
        # train
        targets = np.where(targets == 1, 1, -1)     # targets {-1,1} for delta rule
        np.random.seed(1)
        p_batch = SingleLayerPerceptron(learning_rate=learning_rate, epochs=epochs, method='delta', mode='batch')
        p_batch.learn(p, t)
        np.random.seed(1)
        p_seq = SingleLayerPerceptron(learning_rate=learning_rate, epochs=epochs, method='delta', mode='sequential')
        p_seq.learn(p, t)

        # plot decision boundaries and learning curves
        # p_batch.plot_decision_boundary(plot_db, xlim, r'batch mode $\eta={}$'.format(learning_rate))
        # p_seq.plot_decision_boundary(plot_db, xlim, r'sequential mode $\eta={}$'.format(learning_rate))
        # p_batch.plot_learning_curve(plot_lc, r'batch mode $\eta={}$'.format(learning_rate))
        # p_seq.plot_learning_curve(plot_lc, r'sequential mode $\eta={}$'.format(learning_rate))
#         p_batch.plot_decision_boundary(plot_db, xlim, r'batch mode order {}'.format(i))
#         p_seq.plot_decision_boundary(plot_db, xlim, r'sequential mode order {}'.format(i))
#         p_batch.plot_learning_curve(plot_lc, r'batch mode order {}'.format(i))
#         p_seq.plot_learning_curve(plot_lc, r'sequential mode order {}'.format(i))
#         i += 1
# plot_db.set_title('Batch vs sequential - Delta')
# plot_lc.set_title('Batch vs sequential - Delta')
# plot_lc.set_xticks(np.arange(0, epochs + 1, 2))


###########
# No bias #
###########

epochs = 20
learning_rate = 0.001

# train
targets = np.where(targets == 1, 1, -1)     # targets {-1,1} for delta rule
np.random.seed(1)
p_no_bias = SingleLayerPerceptron(learning_rate=learning_rate, epochs=epochs, method='delta', mode='batch', bias=False)
p_no_bias.learn(patterns, targets)
np.random.seed(1)
p_bias = SingleLayerPerceptron(learning_rate=learning_rate, epochs=epochs, method='delta', mode='batch', bias=True)
p_bias.learn(patterns, targets)

# plot decision boundaries and learning curves
# p_no_bias.plot_decision_boundary(plot_db, xlim, 'without bias weight')
# p_bias.plot_decision_boundary(plot_db, xlim, 'with bias weight')
# p_no_bias.plot_learning_curve(plot_lc, 'without bias weight')
# p_bias.plot_learning_curve(plot_lc, 'with bias weight')
# plot_db.set_title('Bias weight - Delta')
# plot_lc.set_title('Bias weight - Delta')
# plot_lc.set_xticks(np.arange(0, epochs + 1, 2))


###############################
# Non-linearly separable data #
###############################

epochs = 100
learning_rate = 0.001

n_runs = 1     # average results on several runs
accuracy_perceptron = np.zeros((4, 2))
accuracy_delta = np.zeros((4, 2))
for n in range(1, n_runs+1):
    # subsample data
    labels = np.unique(targets)
    sub_patterns = []
    sub_targets = []
    for percentageA, percentageB in [(.75, .75), (.5, 1.), (1., .5)]:
        np.random.seed(n)
        idxA = np.where(targets == labels[1])[1]
        idxB = np.where(targets == labels[0])[1]
        np.random.shuffle(idxA)
        np.random.shuffle(idxB)
        idxA = idxA[:round(percentageA*len(idxA))]
        idxB = idxB[:round(percentageB*len(idxB))]
        idx = np.concatenate((idxA, idxB))
        sub_patterns.append(patterns[:, idx])
        sub_targets.append(targets[:, idx])
    np.random.seed(n)
    idxA1 = np.where((targets[0] == labels[1]) & (patterns[0] < 0))[0]
    idxA2 = np.where((targets[0] == labels[1]) & (patterns[0] > 0))[0]
    np.random.shuffle(idxA1)
    np.random.shuffle(idxA2)
    idxA1 = idxA1[:round(.8*len(idxA1))]
    idxA2 = idxA2[:round(.2*len(idxA2))]
    idxB = np.where(targets == labels[0])[1]
    idx = np.concatenate((idxA1, idxA2, idxB))
    sub_patterns.append(patterns[:, idx])
    sub_targets.append(targets[:, idx])

    i = 0
    legend = ['25% each class', '50% class A', '50% class B', '20% class A left, 80% class A right']
    for p, t in zip(sub_patterns, sub_targets):
        # check subsample (debug)
        # print('class A:', np.where(t == labels[1])[1].shape)
        # print('class A right:', np.where((t[0] == labels[1]) & (p[0] > 0))[0].shape)
        # print('class A left:', np.where((t[0] == labels[1]) & (p[0] < 0))[0].shape)
        # print('class B:', np.where(t == labels[0])[1].shape)
        # print()

        # train
        targets = np.where(targets == 1, 1, 0)  # targets {0,1} for perceptron learning
        np.random.seed(n)
        p_perceptron = SingleLayerPerceptron(learning_rate=learning_rate, epochs=epochs, method='perceptron', mode='batch')
        p_perceptron.learn(p, t)
        targets = np.where(targets == 1, 1, -1)  # targets {-1,1} for delta rule
        np.random.seed(n)
        p_delta = SingleLayerPerceptron(learning_rate=learning_rate, epochs=epochs, method='delta', mode='batch')
        p_delta.learn(p, t)

#         # plot decision boundaries
#         p_perceptron.plot_decision_boundary(plot_db, xlim, 'perceptron ' + legend[i])
#         p_delta.plot_decision_boundary(plot_db, xlim, 'delta ' + legend[i])
#
#         # performance
#         labels = np.unique(targets)
#         idxA = np.where(targets == labels[1])[1]
#         idxB = np.where(targets == labels[0])[1]
#         targets = np.where(targets == 1, 1, 0)  # targets {0,1} for perceptron learning
#         y = p_perceptron.predict(patterns)
#         accuracy_perceptron[i][0] += len(np.where(targets[:, idxA] == y[:, idxA])[1]) / len(idxA)
#         accuracy_perceptron[i][1] += len(np.where(targets[:, idxB] == y[:, idxB])[1]) / len(idxB)
#         targets = np.where(targets == 1, 1, -1)  # targets {-1,1} for delta rule
#         y = p_delta.predict(patterns)
#         accuracy_delta[i][0] += len(np.where(targets[:, idxA] == y[:, idxA])[1]) / len(idxA)
#         accuracy_delta[i][1] += len(np.where(targets[:, idxB] == y[:, idxB])[1]) / len(idxB)
#         i += 1
# for i in range(accuracy_perceptron.shape[0]):
#     print('Accuracy perceptron class A:', accuracy_perceptron[i][0] / n_runs)
#     print('Accuracy perceptron class B:', accuracy_perceptron[i][1] / n_runs)
#     print('Accuracy delta class A:', accuracy_delta[i][0] / n_runs)
#     print('Accuracy delta class B:', accuracy_delta[i][1] / n_runs)
#     print()
# plot_db.set_title('Perceptron vs Delta - subsample')


########################
# Two-layer Perceptron #
########################

learning_rate = 0.001
epochs = 30

# train
np.random.seed(1)
perceptron = TwoLayerPerceptron(learning_rate=learning_rate, epochs=epochs, hidden_nodes=3)
perceptron.learn(patterns, targets)

# plot decision boundaries



# show figures
plot_db.legend(loc='bottom right')
plot_lc.legend(loc='upper right')
plot_db.set_xlabel(r'$x_1$')
plot_db.set_ylabel(r'$x_2$')
plot_lc.set_xlabel('epoch')
plot_lc.set_ylabel('misclassification error')
plot_db.axis(xlim)
fig_db.show()
# fig_lc.show()
