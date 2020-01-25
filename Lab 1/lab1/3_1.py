import numpy as np
import matplotlib.pyplot as plt
from perceptron import SingleLayerPerceptron, TwoLayerPerceptron
from utility import plot_decision_boundary, plot_learning_curve


# Instructions:
# 1. Set learning rate and epochs
# 2. Comment/uncomment plots depending on what you want to see (you can also uncomment all and get lots of plots)

learning_rate = 0.001
epochs = 50


###################
# Data generation #
###################

# generate linearly-separable data
np.random.seed(1)
N = 100
mu = [(-3., -1.5), (0., 0.)]
sigma = [0.5, 0.5]
patterns = np.zeros((2, 2*N))
patterns[0, :N] = np.random.randn(1, N) * sigma[0] + mu[0][0]
patterns[1, :N] = np.random.randn(1, N) * sigma[0] + mu[0][1]
patterns[0, N:] = np.random.randn(1, N) * sigma[1] + mu[1][0]
patterns[1, N:] = np.random.randn(1, N) * sigma[1] + mu[1][1]
targets = np.concatenate((np.ones(N), np.zeros(N)))

# shuffle generated data
np.random.seed(1)
idx = np.arange(2*N)
np.random.shuffle(idx)
patterns = patterns[:, idx]
targets = targets[idx]

# plot generated data
idx = np.where(targets == 0)[0]
plt.scatter(patterns[0, idx], patterns[1, idx], c='r', marker='*')
idx = np.where(targets == 1)[0]
plt.scatter(patterns[0, idx], patterns[1, idx], c='b', marker='.')
xlim = [min(patterns[0])-0.1, max(patterns[0])+0.1, min(patterns[1])-0.1, max(patterns[1])+0.1]


#####################################
# Perceptron learning vs Delta rule #
#####################################

# train
targets = targets.reshape(1, -1)
np.random.seed(1)
p_perceptron = SingleLayerPerceptron(learning_rate, epochs, method='perceptron')
p_perceptron.learn(patterns, targets)
np.random.seed(1)
targets = np.where(targets == 1, 1, -1)     # targets {-1,1} for delta rule
p_delta = SingleLayerPerceptron(learning_rate, epochs, method='delta')
p_delta.learn(patterns, targets)

# plot decision boundaries
# plot_decision_boundary(p_perceptron, xlim, 'm')
# plot_decision_boundary(p_delta, xlim, 'g')
# plt.title('Perceptron learning')
# plt.title('Perceptron learning vs delta rule')
# plt.show()

# plot learning curves
# plot_learning_curve(p_perceptron, 'perceptron learning')
# plot_learning_curve(p_delta, 'delta rule')
# plt.title('Perceptron learning')
# plt.title('Perceptron learning vs delta rule')
# plt.show()


#######################
# Batch vs sequential #
#######################

# train
np.random.seed(1)
p_batch = SingleLayerPerceptron(learning_rate, epochs, method='delta', mode='batch')
p_batch.learn(patterns, targets)
np.random.seed(1)
p_seq = SingleLayerPerceptron(learning_rate, epochs, method='delta', mode='sequential')
p_seq.learn(patterns, targets)

# plot decision boundaries
# plot_decision_boundary(p_batch, xlim, 'm')
# plot_decision_boundary(p_seq, xlim, 'g')
# plt.title('Batch vs sequential')
# plt.show()

# plot learning curves
# plot_learning_curve(p_batch, 'batch mode')
# plot_learning_curve(p_seq, 'sequential mode')
# plt.title('Batch vs sequential')
# plt.show()


###########
# No bias #
###########

# train
np.random.seed(1)
p_no_bias = SingleLayerPerceptron(learning_rate, epochs, method='delta', mode='batch', bias=False)
p_no_bias.learn(patterns, targets)

# plot decision boundaries
plot_decision_boundary(p_no_bias, xlim, 'm')
plt.title('No bias')
plt.show()

# plot learning curves
plot_learning_curve(p_no_bias, 'no bias')
plt.show()
