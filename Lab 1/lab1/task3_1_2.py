from perceptron import SLP
from utility import *
import matplotlib.animation as an


def generate_data(N, mu, sigma, seed=None):
    if seed is not None:
        np.random.seed(seed)
    patterns = np.concatenate((
        np.random.multivariate_normal(mu[0], np.diag(np.ones(2)*sigma[0]), size=N).T,
        np.random.multivariate_normal(mu[1], np.diag(np.ones(2)*sigma[1]), size=N).T
    ), axis=1)
    targets = np.concatenate((np.ones(N), np.zeros(N))).reshape(1, -1)
    return patterns, targets


def animate_decision_boundary(frame, perceptron, line, axes):
    w = perceptron.weights_history[frame][0]
    x1, x2 = get_decision_boundary_slp(w, perceptron.bias, xlim)
    line.set_xdata(x1)
    line.set_ydata(x2)
    line.set_label('decision boundary epoch {}'.format(frame))
    axes.legend()
    return line,


def make_animation_decision_boundary(patterns, targets, xlim, perceptron):
    fig, ax = plt.subplots()
    plot_data(patterns, targets)
    line, = ax.plot(0, 0, 'y')
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(xlim[2], xlim[3])
    animation = an.FuncAnimation(fig, func=animate_decision_boundary, fargs=(perceptron, line, ax), interval=500,
                                 frames=len(perceptron.weights_history))
    ax.legend()
    plt.show()


seed = 1

# generate data
patterns, targets = generate_data(N=100, mu=[(-1, -1), (1, 1)], sigma=[.15, .15], seed=seed)
patterns, targets = shuffle_data(patterns, targets, seed=seed)
xlim = (min(patterns[0, :])-.2, max(patterns[0, :])+.2, min(patterns[1, :])-.2, max(patterns[1, :]+.2))

# train perceptrons
init_weights = np.array([[-.1, .1, .1]])
p1 = SLP(learning_rate=0.0005, n_epochs=20, rule='perceptron', mode='batch', init_weights=init_weights, animation=True)
p2 = SLP(learning_rate=0.0005, n_epochs=20, rule='delta', mode='batch', init_weights=init_weights)
p3 = SLP(learning_rate=0.01, n_epochs=20, rule='perceptron', mode='batch', init_weights=init_weights)
p4 = SLP(learning_rate=0.0049, n_epochs=20, rule='delta', mode='batch', init_weights=init_weights)
p5 = SLP(learning_rate=0.0001, n_epochs=20, rule='delta', mode='batch', init_weights=init_weights)
p6 = SLP(learning_rate=0.01, n_epochs=20, rule='delta', mode='batch', init_weights=init_weights)
p7 = SLP(learning_rate=0.01, n_epochs=20, rule='delta', mode='sequential', init_weights=init_weights)
p8 = SLP(learning_rate=0.0001, n_epochs=20, rule='delta', mode='sequential', init_weights=init_weights)
p9 = SLP(learning_rate=0.0001, n_epochs=20, rule='delta', mode='sequential', init_weights=init_weights)
p10 = SLP(learning_rate=0.001, n_epochs=20, rule='delta', mode='batch', bias=False)
p1.learn(patterns, targets)
p2.learn(patterns, targets)
p3.learn(patterns, targets)
p4.learn(patterns, targets)
p5.learn(patterns, targets)
p6.learn(patterns, targets)
p7.learn(patterns, targets)
p8.learn(patterns, targets)
p9.learn(*shuffle_data(patterns, targets))
p10.learn(patterns, targets)

# plot decision boundary
plot_data(patterns, targets)
# plot_decision_boundary_slp(p1, xlim, label=r'perceptron $\eta={}$'.format(p1.learning_rate))
# plot_decision_boundary_slp(p2, xlim, label=r'delta $\eta={}$'.format(p2.learning_rate))
# plot_decision_boundary_slp(p3, xlim, label=r'perceptron $\eta={}$'.format(p3.learning_rate))
# plot_decision_boundary_slp(p4, xlim, label=r'delta $\eta={}$'.format(p4.learning_rate))
# plot_decision_boundary_slp(p5, xlim, label=r'batch $\eta={}$'.format(p5.learning_rate))
# plot_decision_boundary_slp(p6, xlim, label=r'batch $\eta={}$'.format(p6.learning_rate))
# plot_decision_boundary_slp(p7, xlim, label=r'sequential $\eta={}$'.format(p7.learning_rate))
# plot_decision_boundary_slp(p8, xlim, label=r'sequential $\eta={}$, order 1'.format(p8.learning_rate))
# plot_decision_boundary_slp(p9, xlim, label=r'sequential $\eta={}$, order 2'.format(p9.learning_rate))
# plot_decision_boundary_slp(p10, xlim, label='no bias')
plt.legend()
plt.show()

# plot learning curve
# plot_learning_curve(p1, label=r'perceptron $\eta={}$'.format(p1.learning_rate), style='-o')
# plot_learning_curve(p2, label=r'delta $\eta={}$'.format(p2.learning_rate), style='-o')
# plot_learning_curve(p3, label=r'perceptron $\eta={}$'.format(p3.learning_rate), style='-o')
# plot_learning_curve(p4, label=r'delta $\eta={}$'.format(p4.learning_rate), style='-o')
# plot_learning_curve(p5, label=r'batch $\eta={}$'.format(p5.learning_rate), style='-o')
# plot_learning_curve(p6, label=r'batch $\eta={}$'.format(p6.learning_rate), style='-o')
# plot_learning_curve(p7, label=r'sequential $\eta={}$'.format(p7.learning_rate), style='-o')
# plot_learning_curve(p8, label=r'sequential $\eta={}$, order 1'.format(p8.learning_rate), style='-o')
# plot_learning_curve(p9, label=r'sequential $\eta={}$, order 2'.format(p9.learning_rate), style='-o')
# plot_learning_curve(p10, label='no bias', style='-o')
plt.legend()
plt.show()

# run from shell for this, it does not work in PyCharm
make_animation_decision_boundary(patterns, targets, xlim, p1)
