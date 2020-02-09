import numpy as np
from hebbian_network import Hebbian_Network

# Section 2.2, the original patterns
x1 = [-1, -1, 1, -1, 1, -1, -1, 1]
x2 = [-1,-1,-1, -1, -1, 1, -1, -1]
x3 = [-1, 1, 1, -1, -1, 1, -1, 1]
original_patterns = np.vstack((x1, x2, x3))



# Task 3.1, distorted data
x1d = [1, -1, 1, -1, 1, -1, -1, 1]
x2d = [1, 1, -1, -1, -1, 1, -1, -1]
x3d = [1, 1, 1, -1, 1, 1, -1, 1]
dist_patterns = np.vstack((x1d, x2d, x3d))

hn = Hebbian_Network()
hn.little_model(original_patterns)

result = hn.recall(x1d)

gridsearch = np.zeros((256,8))
for i in range(256):
    bin_list = [int(x) for x in bin(i)[2:]]
    for j in range(len(bin_list)):
        gridsearch[i][8-len(bin_list)+j] = bin_list[j]


gridsearch = np.where(gridsearch == 1, 1, -1)

attractors = []
for i in range(256):
    attractor = hn.recall(gridsearch[i])
    attractors.append(attractor)

unique_attractors = np.unique(attractors,axis=0)


