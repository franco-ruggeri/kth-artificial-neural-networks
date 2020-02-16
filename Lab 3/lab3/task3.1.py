import numpy as np
from hopfieldnetwork import HopfieldNetwork


###############
# Section 2.2 #
###############
x1 = [-1, -1, 1, -1, 1, -1, -1, 1]
x2 = [-1, -1, -1, -1, -1, 1, -1, -1]
x3 = [-1, 1, 1, -1, -1, 1, -1, 1]
original_patterns = np.vstack((x1, x2, x3))
hn = HopfieldNetwork()
hn.learn(original_patterns)

# check if the network stored the patterns
for pattern in original_patterns:
    state = hn.recall(pattern)[0]
    if any(state != pattern):
        print('Error')
        exit(-1)


############
# Task 3.1 #
############
x1d = [1, -1, 1, -1, 1, -1, -1, 1]
x2d = [1, 1, -1, -1, -1, 1, -1, -1]
x3d = [1, 1, 1, -1, 1, 1, -1, 1]
distorted_patterns = np.vstack((x1d, x2d, x3d))
max_iters = round(np.log(original_patterns.shape[1]))   # see lab instructions

# recall from distorted patterns
for i in range(len(distorted_patterns)):
    state = hn.recall(distorted_patterns[i], max_iters=max_iters)[0]
    idx = np.where((original_patterns == state).all(axis=1))[0]
    if len(idx) > 0:
        print('Distorted pattern', i+1, 'converged to stored pattern', idx[0]+1)
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
    attractor = hn.recall(gridsearch[i], max_iters=max_iters)[0]
    if attractor is not None:
        attractors.append(attractor)
attractors = np.unique(attractors, axis=0)
print('Number of attractors:', len(attractors))
print('Attractors:\n', attractors)
