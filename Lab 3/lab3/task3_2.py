import numpy as np
import utility as u

np.random.seed(100)

# load and store pictures
patterns = u.load_pictures()
stored_patterns = patterns[0:3]
hn = u.store_pictures(stored_patterns)

# display stored patterns
for p in stored_patterns:
    u.plot_picture(p)

# complete pattern
for i in [9, 10]:
    recalled = hn.recall(patterns[i], synchronous=True)[0]
    u.plot_picture(patterns[i])
    u.plot_picture(recalled)

# sequential (asynchronous) update
for i in [9, 10]:
    converged = False
    while not converged:
        result = hn.recall(patterns[i], synchronous=False, max_iters=100, plot=True)
        converged = result[2]
        state = result[0]
        u.plot_picture(state, title='Sequential dynamics - Converged')
