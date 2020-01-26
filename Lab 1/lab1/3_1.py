import numpy as np
import matplotlib.pyplot as plt
from perceptron import SingleLayerPerceptron

###################
# Data generation #
###################

# generate linearly-separable data
mu = [(2., .5), (0., 0.)]
sigma = [0.3, 0.3]
np.random.seed(1)
N = 100
patterns = np.zeros((2, 2*N))
patterns[0, :N] = np.random.randn(1, N) * sigma[0] + mu[0][0]
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
targets = targets[idx]
targets = targets.reshape(1, -1)

# prepare plots
fig_db = plt.figure()
plt_db = plt.axes()
fig_lc = plt.figure()
plt_lc = plt.axes()

# plot generated data
idx = np.where(targets[0, :] == 0)[0]
plt_db.scatter(patterns[0, idx], patterns[1, idx], c='r', marker='*', label='class A')
idx = np.where(targets[0, :] == 1)[0]
plt_db.scatter(patterns[0, idx], patterns[1, idx], c='b', marker='.', label='class B')


#####################################
# Perceptron learning vs Delta rule #
#####################################

epochs = 30
for learning_rate in [0.01, 0.001, 0.0001]:
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
#     p_perceptron.plot_decision_boundary(plt_db, xlim, r'perceptron $\eta={}$'.format(learning_rate))
#     p_delta.plot_decision_boundary(plt_db, xlim, r'delta $\eta={}$'.format(learning_rate))
#     p_perceptron.plot_learning_curve(plt_lc, r'perceptron $\eta={}$'.format(learning_rate))
#     p_delta.plot_learning_curve(plt_lc, r'delta $\eta={}$'.format(learning_rate))
# plt_db.set_title('Perceptron vs Delta - batch mode')
# plt_lc.set_title('Perceptron vs Delta - batch mode')
# plt_lc.set_xticks(np.arange(0, epochs + 1, 2))


#######################
# Batch vs sequential #
#######################

# TODO: ripeti questo per 2 volte mischiando il training set, cosi' mostri che uno dipende dall'ordinamento e l'altro no
epochs = 30

for learning_rate in [0.1, 0.001]:
    # train
    targets = np.where(targets == 1, 1, -1)     # targets {-1,1} for delta rule
    np.random.seed(1)
    p_batch = SingleLayerPerceptron(learning_rate=learning_rate, epochs=epochs, method='delta', mode='batch')
    p_batch.learn(patterns, targets)
    np.random.seed(1)
    p_seq = SingleLayerPerceptron(learning_rate=learning_rate, epochs=epochs, method='delta', mode='sequential')
    p_seq.learn(patterns, targets)

    # plot decision boundaries and learning curves
    p_batch.plot_decision_boundary(plt_db, xlim, r'batch mode $\eta={}$'.format(learning_rate))
    p_seq.plot_decision_boundary(plt_db, xlim, r'sequential mode $\eta={}$'.format(learning_rate))
    p_batch.plot_learning_curve(plt_lc, r'batch mode $\eta={}$'.format(learning_rate))
    p_seq.plot_learning_curve(plt_lc, r'sequential mode $\eta={}$'.format(learning_rate))
plt_db.set_title('Batch vs sequential - Delta')
plt_lc.set_title('Batch vs sequential - Delta')
plt_lc.set_xticks(np.arange(0, epochs + 1, 2))


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

# plot decision boundaries and learning curves
# p_no_bias.plot_decision_boundary(plt_db, xlim, 'no bias')
# p_no_bias.plot_learning_curve(plt_lc, 'no bias')
# plt_lc.set_xticks(np.arange(0, epochs + 1, 2))


# show figures
plt_db.set_xlabel(r'$x_1$')
plt_db.set_ylabel(r'$x_2$')
plt_lc.set_xlabel('epoch')
plt_lc.set_ylabel('misclassification error')
plt_db.axis(xlim)
plt_db.legend(loc='upper right')
plt_lc.legend(loc='upper right')
fig_db.show()
fig_lc.show()
