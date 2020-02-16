import numpy as np
import matplotlib.pyplot as plt
from hopfieldnetwork import HopfieldNetwork


data = np.loadtxt('pict.dat', delimiter=",").reshape(-1,1024)

training_data = data[0:3]
test_data = data[3:]

# hn = Hebbian_Network()
# hn.little_model(training_data)
#
# #Calculate energy for attractors
# for i in range(training_data.shape[0]):
#     print("Energy at attractor {}: {}".format(i+1,hn.energy(training_data[i])))
#
# for i in range(test_data.shape[0]):
#     print("Energy at distorted picture {}: {}".format(i+4,hn.energy(test_data[i])))
#
# p10_recalled, iterations, energy = hn.recall(test_data[-2], check_convergence=True)

#print(iterations)
#plt.plot(energy)
#plt.show()

# Random weight matrix
hn_2 = HopfieldNetwork()
dimension = 1024
W = np.random.normal(0,1,(1024,1024))
hn_2.weights = W
start_state = np.random.rand(dimension)


recalled_start, e1 = hn_2.recall(start_state, max_iters=200)

plt.plot(e1)
plt.show()



# Symmetric weight matrix

hn_3 = HopfieldNetwork()
Wsym = 0.5*(W+np.transpose(W))
hn_3.weights = Wsym

recalled, e2 = hn_3.recall(start_state, max_iters=200)
plt.plot(e2)
plt.show()


