import numpy as np
import random
import matplotlib.pyplot as plt
from perceptron import PCT

from scipy.stats import multivariate_normal

n = 100
mA = np.array([ 6.0, 1.5])
mB = np.array([-1.0, -1.5])
sigmaA = 0.5
sigmaB = 0.5
covA = [[sigmaA,0],[0,sigmaA]]
covB = [[sigmaB,0],[0,sigmaB]]

x1, y1 = np.random.multivariate_normal(mA, covA, 100).T
x2, y2 = np.random.multivariate_normal(mB, covB, 100).T


classA = np.concatenate((x1.reshape(1, 100), y1.reshape(1, 100)), axis=0)
targetsA = np.ones(100)

classB = np.concatenate((x2.reshape(1, 100), y2.reshape(1, 100)), axis=0)
targetsB = -1*np.ones(100)

patterns = np.concatenate((classA,classB),axis=1)
targets = np.concatenate((targetsA,targetsB))
#print(patterns.shape)
#print(targets.shape)
data = np.concatenate((patterns,targets.reshape(1,200)))
np.random.shuffle(data.T)

patterns = data[0:2,]

targets = data[2,]
np.random.shuffle(targets)

pct = PCT(patterns)
#pct.delta_rule(patterns, targets)
#print(pct.get_weights())
pct.perceptron_learning(patterns, targets)
print(pct.get_weights())

def decision_boundary(x, w):

    w1, w2, w3 = w[0], w[1], w[2]

    return -(w1/w2)*x - w3/w2

############## decision boundary ##############

x = np.linspace(0, 5, 100)

decision_line = decision_boundary(x, pct.get_weights())


plt.plot(x, decision_line, label="decision line")
plt.plot(classA[0,], classB[1,], 'o', label='class A')
plt.plot(classB[0,], classB[1,], 'o', label='class B')
plt.legend()
plt.show()

