import numpy as np
import matplotlib.pyplot as plt
from hopfieldnetwork import HopfieldNetwork

data = np.loadtxt('datasets/pict.dat', delimiter=",").reshape(-1,1024)


def generate_random_patterns(nPatterns, dimension, activity, bias=False):

    if bias:
        random_data = np.random.randn(nPatterns, dimension) + 0.5
    else:
        random_data = np.random.randn(nPatterns, dimension)

    random_patterns = np.where(random_data < activity, 1, 0)
    return random_patterns

def assignment3_6_1():

    thetas = np.arange(0, 3, 0.1)
    n_patterns = np.arange(1,10)
    dimension = 100
    bias_pattern_matrix = np.zeros((len(thetas),len(n_patterns)))
    for idx, theta in enumerate(thetas):

        for jdx, pattern in enumerate(n_patterns):
            training_set = generate_random_patterns(pattern, dimension, activity=0.1)
            count = 0
            for i in range(pattern):
                hn = HopfieldNetwork()
                hn.learn(training_set[0:i], sparse=True, activity=0.1, self_connections=True)
                recall = hn.recall(training_set[i-1], update_rule='sparse', theta=theta)[0]
                if np.array_equal(recall, training_set[i-1]):
                    count += 1
            bias_pattern_matrix[idx][jdx] = count

    print("Rows are the thetas and columns are the number of patterns and the elements how many correctly remembered patterns:")
    print(bias_pattern_matrix)

assignment3_6_1()
