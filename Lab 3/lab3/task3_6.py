import numpy as np
import utility as u
import matplotlib.pyplot as plt
from hopfieldnetwork import HopfieldNetwork


def generate_random_patterns(n_patterns, dimension, activity):
    # choice idx of 1s, the number is such that the activity is the wanted one
    idx = np.random.choice(n_patterns * dimension, size=int(round(activity*n_patterns*dimension)), replace=False)

    # fill 1s in the right positions
    patterns = np.zeros((n_patterns, dimension))
    for i in idx:
        row = int(i / dimension)
        col = i % dimension
        patterns[row, col] = 1
    return patterns


np.random.seed(1)
patterns = u.load_pictures()

thetas = np.arange(0, 3.5, 0.5)
dimension = 100
activities = [0.1, 0.05, 0.01]
n_patterns = np.arange(1, 101)

for activity in activities:
    for theta in thetas:
        convergence = np.zeros(len(n_patterns))
        for i, n in enumerate(n_patterns):
            np.random.seed(1)
            patterns = generate_random_patterns(n, dimension, activity=activity)
            hn = HopfieldNetwork(sparse=True, bias=theta)
            hn.learn(patterns)

            # check stability of patterns
            for pattern in patterns:
                # stability of pure pattern
                recall = hn.recall(pattern, synchronous=True, max_iters=1)[0]
                if np.array_equal(recall, pattern):
                    convergence[i] += 1
            convergence[i] /= n
        plt.plot(n_patterns, convergence, label=r'$\theta$={:.2f}'.format(theta))
        plt.title('Activity {}'.format(activity))
        plt.xticks(n_patterns)

    plt.xlabel('# patterns')
    plt.ylabel('% convergence')
    plt.xticks(np.arange(10, n_patterns[-1], 10))
    plt.legend()
    plt.show()
