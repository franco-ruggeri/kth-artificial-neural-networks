import numpy as np
import matplotlib.pyplot as plt
from hopfieldnetwork import HopfieldNetwork

data = np.loadtxt('datasets/pict.dat', delimiter=",").reshape(-1,1024)

def add_noise(x, percentage):

    noisy_x = np.array(x)
    shuffle_size = int(percentage*x.shape[0])
    positions = np.random.choice(x.shape[0], shuffle_size, replace=False)
    extracted_positions = noisy_x[positions]
    np.random.shuffle(extracted_positions)
    noisy_x[positions] = extracted_positions

    return noisy_x

def misclassification(original_pic, recalled_pic):

    correct_pixels = np.where(recalled_pic == original_pic)
    classification_ratio = np.shape(correct_pixels)[1]/np.shape(original_pic)[0]
    return classification_ratio

def generate_random_patterns(nPatterns, dimension, bias=False):

    if bias:
        random_data = np.random.randn(nPatterns, dimension) + 0.5
    else:
        random_data = np.random.randn(nPatterns, dimension)

    random_patterns = np.where(random_data > 0, 1, -1)
    return random_patterns

def assignment3_5_1():

    n_patterns = [3,4,5,6,7]
    ratios = []
    for i in n_patterns:
        hn = HopfieldNetwork()
        training_set = data[0:i]
        test_set = data[i:]
        hn.learn(training_set)
        dist_pic = add_noise(training_set[0], 0.1)
        recall = hn.recall(dist_pic, synchronous=True)[0]
        ratios.append(misclassification(training_set[0], recall))

    plt.plot(n_patterns, ratios)
    plt.xlabel("Number of patterns")
    plt.ylabel("Accuracy")
    plt.title("Decrease in performance as a function of the number of patterns")
    plt.show()

#assignment3_5_1()

def assignment3_5_2():

    n_patterns = np.arange(1,200)
    dimension = 1024
    random_data = generate_random_patterns(len(n_patterns), dimension)
    ratios = []
    for i in n_patterns:
        hn = HopfieldNetwork()
        training_set = random_data[0:i]
        test_set = random_data[i:]
        hn.learn(training_set)
        dist_pic = add_noise(training_set[0], 0.1)
        recall = hn.recall(dist_pic, synchronous=True)[0]
        ratios.append(misclassification(training_set[0], recall))

    plt.plot(n_patterns, ratios)
    plt.xlabel("Number of patterns")
    plt.ylabel("Accuracy")
    plt.title("Decrease in performance with random patterns")
    plt.show()

#assignment3_5_2()

def assignment3_5_4():

    n_patterns = np.arange(1,301)
    dimension = 100
    random_data = generate_random_patterns(len(n_patterns), dimension)
    counter = []
    for i in n_patterns:
        hn = HopfieldNetwork()
        hn.learn(random_data[0:i])
        count = 0
        for j in range(np.shape(random_data[0:i])[0]):
            recall = hn.recall(random_data[j], synchronous=True)[0]
            if np.array_equal(recall, random_data[j]):
                count += 1

        counter.append(count / np.shape(random_data[0:i])[0])

    plt.plot(n_patterns, counter)
    plt.xlabel("Number of patterns")
    plt.ylabel("Ability to store earlier patterns")
    plt.show()

#assignment3_5_4()

def assignment3_5_5():

    n_patterns = np.arange(1,301)
    dimension = 100
    random_data = generate_random_patterns(len(n_patterns), dimension)
    counter = []
    for i in n_patterns:
        hn = HopfieldNetwork()
        hn.learn(random_data[0:i])
        count = 0
        for j in range(np.shape(random_data[0:i])[0]):
            dist_pattern = add_noise(random_data[j], 0.1)

            recall = hn.recall(dist_pattern, synchronous=True)[0]
            if np.array_equal(recall, random_data[j]):
                count += 1

        counter.append(count / np.shape(random_data[0:i])[0])

    plt.plot(n_patterns, counter)
    plt.xlabel("Number of patterns")
    plt.ylabel("Ability to store earlier patterns")
    plt.show()


#assignment3_5_5()


def assignment3_5_6():

    n_patterns = np.arange(1,301)
    dimension = 100
    random_data = generate_random_patterns(len(n_patterns), dimension)
    dist_counter = []
    pure_counter = []
    for i in n_patterns:
        hn = HopfieldNetwork()
        hn.learn(random_data[0:i], remove_self_connections=True)
        dist_count = 0
        pure_count = 0
        for j in range(np.shape(random_data[0:i])[0]):
            dist_pattern = add_noise(random_data[j], 0.1)

            dist_recall = hn.recall(dist_pattern, synchronous=True, max_iters=50)[0]
            pure_recall = hn.recall(random_data[j], synchronous=True, max_iters=50)[0]

            if np.array_equal(dist_recall, random_data[j]):
                dist_count += 1

            if np.array_equal(pure_recall, random_data[j]):
                pure_count += 1



        dist_counter.append(dist_count / np.shape(random_data[0:i])[0])
        pure_counter.append(pure_count / np.shape(random_data[0:i])[0])

    plt.plot(n_patterns, dist_counter, label="Noisy pattern")
    plt.plot(n_patterns, pure_counter, label="Pure pattern")
    plt.xlabel("Number of patterns")
    plt.ylabel("Ability to store earlier patterns")
    plt.title("Self connections removed")
    plt.legend()
    plt.show()

#assignment3_5_6()

def assignment3_5_7():

    n_patterns = np.arange(1,301)
    dimension = 100
    random_data = generate_random_patterns(len(n_patterns), dimension, bias=True)
    dist_counter = []
    pure_counter = []
    for i in n_patterns:
        hn = HopfieldNetwork()
        hn.learn(random_data[0:i], remove_self_connections=True)
        dist_count = 0
        pure_count = 0
        for j in range(np.shape(random_data[0:i])[0]):
            dist_pattern = add_noise(random_data[j], 0.1)

            dist_recall = hn.recall(dist_pattern, synchronous=True, max_iters=50)[0]
            pure_recall = hn.recall(random_data[j], synchronous=True, max_iters=50)[0]

            if np.array_equal(dist_recall, random_data[j]):
                dist_count += 1

            if np.array_equal(pure_recall, random_data[j]):
                pure_count += 1



        dist_counter.append(dist_count / np.shape(random_data[0:i])[0])
        pure_counter.append(pure_count / np.shape(random_data[0:i])[0])

    plt.plot(n_patterns, dist_counter, label="Noisy pattern")
    plt.plot(n_patterns, pure_counter, label="Pure pattern")
    plt.xlabel("Number of patterns")
    plt.ylabel("Ability to store earlier patterns")
    plt.title("Self connections removed, with added bias towards 1")
    plt.legend()
    plt.show()


assignment3_5_7()

