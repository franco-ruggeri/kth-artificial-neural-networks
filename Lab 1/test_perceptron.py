import numpy as np
import random
from perceptron import PCT
from scipy.stats import multivariate_normal

n = 100
mA = np.array([ 1.0, 0.5])
mB = np.array([-1.0, 0.0])
sigmaA = 0.5
sigmaB = 0.5
covA = [[sigmaA,0],[0,sigmaA]]
covB = [[sigmaB,0],[0,sigmaB]]

x1, y1 = np.random.multivariate_normal(mA, covA, 100).T
x2, y2 = np.random.multivariate_normal(mB, covB, 100).T


classA = np.concatenate((x1.reshape(1, 100), y1.reshape(1, 100)), axis=0)
print(classA.shape)
targetsA = np.ones(100)

classB = np.concatenate((x2.reshape(1, 100), y2.reshape(1, 100)), axis=0)
print(classB.shape)
targetsB = -1*np.ones(100)

patterns = np.concatenate((classA,classB),axis=1)
targets = np.concatenate((targetsA,targetsB))
print(patterns.shape)
print(targets.shape)
data = np.concatenate((patterns,targets.reshape(1,200)))
np.random.shuffle(data.T)

patterns = data[0:2,]
targets = data[2,]

pct = PCT(patterns)
pct.delta_rule(patterns, targets)
print(pct.show_weights())

