import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(perceptron, xlim, color, N=100):
    x1 = np.linspace(xlim[0], xlim[1], N)
    x2 = np.linspace(xlim[2], xlim[3], N)
    xx1, xx2 = np.meshgrid(x1, x2)
    if perceptron.bias:
        X = np.concatenate((np.ravel(xx1), np.ravel(xx2), np.ones(N * N)), axis=0).reshape(3, N * N)
    else:
        X = np.concatenate((np.ravel(xx1), np.ravel(xx2)), axis=0).reshape(2, N * N)
    y = perceptron.weights @ X
    y = y.reshape(N, N)
    plt.contour(xx1, xx2, y, levels=(0,), colors=color)


def plot_learning_curve(perceptron, label):
    plt.plot(perceptron.mce, 'o-', label=label)
    plt.xticks(np.arange(0, len(perceptron.mce)+1, 2))
    plt.xlabel('epoch')
    plt.ylabel('misclassification error')
