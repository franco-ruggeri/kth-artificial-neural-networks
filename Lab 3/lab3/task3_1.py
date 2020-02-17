import numpy as np
import utility as u
from hopfieldnetwork import HopfieldNetwork


###############
# Section 2.2 #
###############
x1 = [-1, -1, 1, -1, 1, -1, -1, 1]
x2 = [-1, -1, -1, -1, -1, 1, -1, -1]
x3 = [-1, 1, 1, -1, -1, 1, -1, 1]
original_patterns = np.vstack((x1, x2, x3))
hn = HopfieldNetwork(self_connections=True)
hn.learn(original_patterns)
if not u.check_stability(hn, original_patterns):
    print('Error, patterns not stored.')
    exit(-1)


############
# Task 3.1 #
############
x1d = [1, -1, 1, -1, 1, -1, -1, 1]
x2d = [1, 1, -1, -1, -1, 1, -1, -1]
x3d = [1, 1, 1, -1, 1, 1, -1, 1]
distorted_patterns = np.vstack((x1d, x2d, x3d))
max_iters = 100

# recall from distorted patterns
for i in range(len(distorted_patterns)):
    result = hn.recall(distorted_patterns[i], synchronous=True, max_iters=max_iters)
    if result[2]:
        state = result[0]
        if np.array_equal(original_patterns[i], state):
            print('Distorted pattern', i+1, 'correctly converged to stored pattern', i+1)
        else:
            print('Distorted pattern', i+1, 'converged to another state')
    else:
        print('Distorted pattern', i+1, 'not converged to any stored pattern')

# generate all possible patterns with 8 features
gridsearch = np.zeros((256, 8))
for i in range(256):
    bin_list = [int(x) for x in bin(i)[2:]]
    for j in range(len(bin_list)):
        gridsearch[i][8-len(bin_list)+j] = bin_list[j]
gridsearch = np.where(gridsearch == 1, 1, -1)

# search attractors
attractors = []
for i in range(256):
    result = hn.recall(gridsearch[i], synchronous=True, max_iters=max_iters)
    if result[2]:
        attractor = result[0]
        attractors.append(attractor)
attractors = np.unique(attractors, axis=0)
print('Number of attractors:', len(attractors))
print('Attractors:\n', attractors)
