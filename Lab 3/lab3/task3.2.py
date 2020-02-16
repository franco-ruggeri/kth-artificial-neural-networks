import matplotlib.pyplot as plt
import numpy as np
from hebbian_network import Hebbian_Network





data = np.loadtxt('pict.dat', delimiter=",").reshape(-1,1024)

training_data = data[0:3]
test_data = data[3:]

hn = Hebbian_Network()
hn.little_model(training_data)

# check stability
def check_stability(training_data):

    converged_patterns = []
    for pattern in training_data:
        p, iterations = hn.recall(pattern, check_convergence=True)
        converged_patterns.append(iterations)

    return converged_patterns

def plot_images(pattern):

    pattern = pattern.reshape(32,32)
    plt.imshow(pattern)
    plt.show()


# p10 = test_data[-2]
# new_p10 = hn.recall(p10)[0]
#
# pic1 = plot_images(training_data[0])
# pic2 = plot_images(p10)
# pic3 = plot_images(new_p10)


# p11 = test_data[-1]
# recalled_p11 = hn.recall(p11)[0]
#
# pic1 = plot_images(training_data[1])
# pic2 = plot_images(training_data[2])
# pic3 = plot_images(p11)
# pic4 = plot_images(recalled_p11)

p11 = test_data[-1]
new_p11, iterartions = hn.recall(p11, check_convergence=True)

print(iterartions)


