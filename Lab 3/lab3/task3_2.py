import numpy as np
import utility as u

np.random.seed(100)

# load and store pictures
patterns = u.load_pictures()
stored_patterns = patterns[0:3]
hn = u.store_pictures(stored_patterns)

# display stored patterns
for p in stored_patterns:
    u.plot_image(p)

# complete pattern
for i in [9, 10]:
    recalled = hn.recall(patterns[i], update_rule='synch')[0]
    u.plot_image(patterns[i])
    u.plot_image(recalled)

# sequential (asynchronous) update
for i in [9, 10]:
    converged = False
    while not converged:
        result = hn.recall(patterns[i], update_rule='asynch', max_iters=100, plot=True)
        converged = result[2]
        state = result[0]
        u.plot_image(state, title='Sequential dynamics - Converged')
