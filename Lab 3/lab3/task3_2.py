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

# display stored patterns
for p in stored_patterns:
    u.plot_image(p)

# complete pattern
for i in [9, 10]:
    recalled = hn.recall(patterns[i], synchronous=True)[0]
    u.plot_image(patterns[i])
    u.plot_image(recalled)

# sequential (asynchronous) update
for i in [9, 10]:
    converged = False
    while not converged:
        result = hn.recall(patterns[i], synchronous=False, max_iters=100, plot=True)
        converged = result[2]
        state = result[0]
        u.plot_image(state, title='Sequential dynamics - Converged')
