from utility import *
from matplotlib import cm
from perceptron import MLP
from mpl_toolkits import mplot3d


def plot_function(x, y, z):
    ax = plt.gca(projection='3d')
    ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)


def make_animation_decision_boundary(patterns, targets, xlim, perceptron):
    # TODO: modifica questo, devo fare l'animazione della funzione

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
x = np.arange(-5, 5, 0.5)
y = np.arange(-5, 5, 0.5)
xx, yy = np.meshgrid(x, y)
z = np.exp(- (xx**2 + yy**2) / 10) - 0.5
patterns = np.concatenate((xx.reshape(1, -1), yy.reshape(1, -1)), axis=0)
targets = z.reshape(1, -1)
grid_size = len(x)

# plot function
plot_function(x, y, z)
plt.show()

# train
perceptron = MLP(learning_rate=0.1, n_epochs=1000, n_hidden=10, mode='batch', animation=True, seed=seed)
perceptron.learn(patterns, targets)

# analyze
z = perceptron.predict(patterns, thresholded=False).reshape(grid_size, grid_size)
plot_function(x, y, z)
plt.show()
plot_learning_curve(perceptron, measure='mse')
plt.show()
