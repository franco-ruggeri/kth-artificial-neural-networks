from utility import *
from perceptron import MLP

seed = 100

# generate data
n_inputs = 8
patterns = np.identity(n_inputs)
patterns = np.where(patterns == 0, -1, patterns)

# train
n_hidden = 3
targets = patterns
perceptron = MLP(learning_rate=0.1, n_epochs=10000, n_hidden=n_hidden, mode='batch', seed=seed)
perceptron.learn(patterns, targets)

# plot learning curve
plot_learning_curve(perceptron)
plt.show()

# inspect internal code
print('Patterns')
print(patterns)
print()
patterns_ = np.concatenate((patterns, np.ones((1, patterns.shape[1]))), axis=0)
print('Activations of hidden neurons')
print((perceptron.W @ patterns_).T)
print()
print('Activations of hidden neurons (thresholded)')
print(np.where(perceptron.W @ patterns_ > 0, 1, -1).T)
print()
print('Weight matrix of first layer')
print(perceptron.W)
