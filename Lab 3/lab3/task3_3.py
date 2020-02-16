import numpy as np
import utility as u
from hopfieldnetwork import HopfieldNetwork

np.random.seed(100)

# load data
patterns = np.loadtxt('datasets/pict.dat', delimiter=",").reshape(-1, 1024)

# store first 3 images
stored_patterns = patterns[0:3]
hn = HopfieldNetwork()
hn.learn(stored_patterns)
if not u.check_stability(hn, stored_patterns):
    print('Error, patterns not stored.')
    exit(-1)

# energy at attractors
for i in range(len(stored_patterns)):
    print('Energy at attractor of pattern {}: {}'.format(i+1, hn.compute_energy(patterns[i])))

# energy at attractors of and at distorted patterns
for i in [9, 10]:
    result = hn.recall(patterns[i], update_rule='asynch')
    energy = result[1]
    print('Energy at attractor of pattern {}: {}'.format(i + 1, energy[-1]))
    print('Energy at distorted pattern {}: {}'.format(i+1, hn.compute_energy(patterns[i])))
    u.plot_energy(energy)


#print(iterations)
#plt.plot(energy)
#plt.show()

# Random weight matrix
# hn_2 = HopfieldNetwork()
# dimension = 1024
# W = np.random.normal(0,1,(1024,1024))
# hn_2.weights = W
# start_state = np.random.rand(dimension)
#
#
# recalled_start, e1 = hn_2.recall(start_state, max_iters=200)
#
# plt.plot(e1)
# plt.show()



# Symmetric weight matrix

# hn_3 = HopfieldNetwork()
# Wsym = 0.5*(W+np.transpose(W))
# hn_3.weights = Wsym
#
# recalled, e2 = hn_3.recall(start_state, max_iters=200)
# plt.plot(e2)
# plt.show()


