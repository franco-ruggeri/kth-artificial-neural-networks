import numpy as np
import matplotlib.pyplot as plt
from hopfieldnetwork import HopfieldNetwork

data = np.loadtxt('datasets/pict.dat', delimiter=",").reshape(-1,1024)

training_data = data[0:3]
test_data = data[3:]
test_arr = np.arange(0,10)

def add_noise(x, percentage):

    noisy_x = np.array(x)
    shuffle_size = int(percentage*x.shape[0])
    positions = np.random.choice(x.shape[0], shuffle_size, replace=False)
    extracted_positions = noisy_x[positions]
    np.random.shuffle(extracted_positions)
    noisy_x[positions] = extracted_positions

    return noisy_x


hn = HopfieldNetwork()
hn.learn(training_data)
print(training_data.shape)

#noise_levels = np.arange(1, 6)*0.10
noise_levels = np.arange(1,5,0.10)*0.10

amount_recovered = np.zeros((training_data.shape[1], noise_levels.shape[0]))

for i in range(training_data.shape[0]):
    for j, noise in enumerate(noise_levels):

        distort_pic = add_noise(training_data[i],noise)
        recall_pic = hn.recall(distort_pic)[0]
        check_recovery = np.where(recall_pic == training_data[i])
        if np.array_equal(recall_pic, training_data[i]):
            amount_recovered[i, j] = 1
        else:
            amount_recovered[i, j] = 0
        print("Noise: {}, Pixels recovered: {}".format(noise,len(check_recovery[0])))


plt.plot(noise_levels, amount_recovered[0], label="p1")
plt.plot(noise_levels, amount_recovered[1], label="p2")
plt.plot(noise_levels, amount_recovered[2], label="p3")
plt.ylabel("Accuracy")
plt.xlabel("Noise levels")
plt.legend()
plt.show()

