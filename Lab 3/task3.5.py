import numpy as np
import matplotlib.pyplot as plt
from hebbian_network import Hebbian_Network

data = np.loadtxt('pict.dat', delimiter=",").reshape(-1,1024)

def add_noise(x, percentage):

    noisy_x = np.array(x)
    shuffle_size = int(percentage*x.shape[0])
    positions = np.random.choice(x.shape[0], shuffle_size, replace=False)
    extracted_positions = noisy_x[positions]
    np.random.shuffle(extracted_positions)
    noisy_x[positions] = extracted_positions

    return noisy_x

def misclassification(x, y):

    recalled = hn.recall(y)[0]
    correct_pixels = np.where(recalled == x)
    print(np.shape(correct_pixels)[1], np.shape(x)[0])
    classification_ratio = np.shape(correct_pixels)[1]/np.shape(x)[0]
    #print(np.shape(correct_pixels)[1], np.shape(x)[0])
    return classification_ratio

def generate_random_patterns(nPatterns):

    random_data = np.random.randn(nPatterns,100) + 0.5
    random_patterns = np.where(random_data > 0, 1, -1)
    return random_patterns



# Part 3.5.1
"""
hn = Hebbian_Network()

nPatterns = 7
training_set = data[0:nPatterns]
test_set = data[nPatterns:]
hn.little_model(training_set)

dist_pic = add_noise(training_set[0], 0.5)

recalled = hn.recall(dist_pic)[0]
"""

# Part 3.5.2
"""
hn2 = Hebbian_Network()

random_data = random_patterns(300)
random_training = random_data[0:300]

hn2.little_model(random_training)

dist2_pic = add_noise(random_training[250], 0.10)
recalled2 = hn2.recall(dist2_pic)[0]

"""

# Part 3.5.4-7


random_training = generate_random_patterns(300)

#ratio_list = []
nPatterns = np.arange(1,301)

counter = []
for i in range(1,301):

    hn = Hebbian_Network()
    hn.little_model(random_training[0:i])
    count = 0
    for j in range(np.shape(random_training[0:i])[0]):
        #dist_pattern = add_noise(random_training[j],0.10) # For 3.5.5
        #recall = hn.recall(dist_pattern)[0] # For 3.5.5
        recall = hn.recall(random_training[j], max_iters=50)[0] #3.5.4
        if np.array_equal(recall, random_training[j]):
            count += 1
        #ratio.append(np.shape(correct_data)[1]/np.shape(random_training[j])[0])
    counter.append(count/np.shape(random_training[0:i])[0])
    #ratio_list.append(np.average(ratio))


plt.plot(nPatterns, counter)
plt.show()



