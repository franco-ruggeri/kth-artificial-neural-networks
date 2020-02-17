import numpy as np
import matplotlib.pyplot as plt
import utility as u

np.random.seed(100)

# load and store pictures
patterns = u.load_pictures()
stored_patterns = patterns[0:3]
hn = u.store_pictures(stored_patterns)

# energy at attractors
for i in range(len(stored_patterns)):
    print('Energy at attractor of picture {}: {}'.format(i + 1, hn.compute_energy(patterns[i])))

# energy at attractors of and at distorted patterns
for i in [9, 10]:
    result = hn.recall(patterns[i], update_rule='asynch')
    energy = result[1]
    print('Energy at attractor of picture {}: {}'.format(i + 1, energy[-1]))
    print('Energy at distorted picture {}: {}'.format(i + 1, hn.compute_energy(patterns[i])))
    u.plot_energy(energy, label='distorted p{}'.format(i + 1))
plt.show()

# random state
state = np.random.choice([-1, 1], size=hn.n_neurons)

# random weight matrix
hn.weights = np.random.randn(hn.n_neurons, hn.n_neurons)
result = hn.recall(state, update_rule='asynch')
# u.plot_image(state)
# u.plot_image(result[0])
u.plot_energy(result[1], label='asymmetric random weight matrix')

# symmetric weight matrix
hn.weights = 0.5 * hn.weights + hn.weights.T
result = hn.recall(state, update_rule='asynch')
# u.plot_image(result[0])
u.plot_energy(result[1], label='symmetric random weight matrix')
plt.show()
