from utility import *
from matplotlib import cm
from perceptron import MLP
from mpl_toolkits import mplot3d
import matplotlib.animation as an


def plot_function(x, y, z, zlim):
    ax = plt.gca(projection='3d')
    ax.plot_surface(x, y, z, cmap=cm.plasma, linewidth=0, antialiased=False)
    ax.set_zlim(zlim[0], zlim[1])


def animate_function(frame, x, y, perceptron, axes, plot, zlim):
    W = perceptron.W_history[frame]
    V = perceptron.V_history[frame]

    patterns = np.concatenate((xx.reshape(1, -1), yy.reshape(1, -1)), axis=0)
    patterns = np.concatenate((patterns, np.ones((1, patterns.shape[1]))), axis=0)

    H = perceptron._activate(W @ patterns)
    H = np.concatenate((H, np.ones((1, patterns.shape[1]))), axis=0)  # add row for bias term
    z = perceptron._activate(V @ H)
    z = z.reshape(grid_size, grid_size)

    if len(plot) == 0:  # first time
        plot.append(axes.plot_surface(x, y, z, cmap=cm.plasma, linewidth=0, antialiased=False))
    else:
        plot[0].remove()
        plot[0] = axes.plot_surface(x, y, z, cmap=cm.plasma, linewidth=0, antialiased=False)
    axes.set_zlim(zlim[0], zlim[1])


def make_animation_function(x, y, perceptron, zlim):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot = []
    animation = an.FuncAnimation(fig, func=animate_function, fargs=(x, y, perceptron, ax, plot, zlim), interval=10,
                                 frames=perceptron.n_epochs+1)
    plt.show()


seed = 1

# generate data
x = np.arange(-5, 5, 0.5)
y = np.arange(-5, 5, 0.5)
xx, yy = np.meshgrid(x, y)
z = np.exp(- (xx**2 + yy**2) / 10) - 0.5
zlim = (np.min(z) - .2, np.max(z) + .2)
patterns = np.concatenate((xx.reshape(1, -1), yy.reshape(1, -1)), axis=0)
targets = z.reshape(1, -1)
grid_size = len(x)

# plot function
plot_function(x, y, z, zlim)
plt.show()

for n_hidden in range(2, 10, 2):
    # train
    perceptron = MLP(learning_rate=0.1, n_epochs=10000, n_hidden=n_hidden, mode='batch', animation=True, seed=seed)
    perceptron.learn(patterns, targets)

    # plot function
    plt.figure()
    z = perceptron.predict(patterns, thresholded=False).reshape(grid_size, grid_size)
    plot_function(x, y, z, zlim)
    plt.title('n_hidden={}'.format(n_hidden))

    # plot learning curve
    plt.figure(2)
    plot_learning_curve(perceptron, measure='mse', label='n_hidden={}'.format(n_hidden))
plt.legend()
plt.show()

# run from shell for this, it does not work in PyCharm
# make_animation_function(x, y, perceptron, zlim)
