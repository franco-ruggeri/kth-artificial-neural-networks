import numpy as np
import utility as u
import itertools as it
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

thetas = np.arange(0, 10, 0.1)
dimension = 100
activities = [0.1, 0.05, 0.01]
bias_pattern_matrix = np.zeros(len(thetas))

print('Row: theta\nCol: # patterns\nValue: # patterns stored')
for activity in activities:
    for i, theta in enumerate(thetas):
        for n in it.count(1):
            np.random.seed(1)
            patterns = generate_random_patterns(n, dimension, activity=activity)
            hn = HopfieldNetwork(sparse=True, bias=theta)
            hn.learn(patterns)

            all_stored = True
            for pattern in patterns:
                recall = hn.recall(pattern)[0]
                if not np.array_equal(recall, pattern):
                    all_stored = False
            if all_stored:
                bias_pattern_matrix[i] = n
            else:
                break   # capacity reached

    print()
    print('Activity:', activity)
    print(bias_pattern_matrix)
