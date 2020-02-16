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
    print('Energy at attractor of picture {}: {}'.format(i + 1, hn.compute_energy(patterns[i])))

# energy at attractors of and at distorted patterns
for i in [9, 10]:
    result = hn.recall(patterns[i], update_rule='asynch')
    energy = result[1]
    print('Energy at attractor of picture {}: {}'.format(i + 1, energy[-1]))
    print('Energy at distorted picture {}: {}'.format(i + 1, hn.compute_energy(patterns[i])))
    u.plot_energy(energy, title='Energy for distorted picture {}'.format(i))
    print(energy)

# random state
state = np.random.choice([-1, 1], size=hn.n_neurons)

# random weight matrix
hn.weights = np.random.randn(hn.n_neurons, hn.n_neurons)
result = hn.recall(state)
# u.plot_image(state)
# u.plot_image(result[0])
u.plot_energy(result[1], title='Energy with asymmetric random weight matrix')

# symmetric weight matrix
hn.weights = 0.5 * hn.weights + hn.weights.T
result = hn.recall(state)
# u.plot_image(result[0])
u.plot_energy(result[1], title='Energy with symmetric random weight matrix')
