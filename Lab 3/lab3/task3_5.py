import numpy as np
import matplotlib.pyplot as plt
import utility as u
from hopfieldnetwork import HopfieldNetwork
from hopfieldnetwork import sign


def measure_convergence(original_pic, recalled_pic):
    """Measure convergence using Hamming distance."""
    correct_pixels = np.where(recalled_pic == original_pic)[0]
    return len(correct_pixels) / len(original_pic)


def capacity_pictures():
    n_patterns = np.arange(1, 8)
    noise_perc = 0.1
    ratios = []

    hn = HopfieldNetwork(self_connections=False)
    for n in n_patterns:
        # learn another pattern (Hebbian learning is incremental!)
        stored_patterns = patterns[0:n]
        hn.learn(stored_patterns[-1].reshape(1, -1))

        # check convergence of distorted patterns
        ratios.append(0)
        for pattern in stored_patterns:
            dist_pattern = u.add_noise(pattern, noise_perc)
            recall = hn.recall(dist_pattern, synchronous=True)[0]
            ratios[-1] += measure_convergence(pattern, recall)
        ratios[-1] /= n

    plt.plot(n_patterns, ratios, '-s')
    plt.grid()
    plt.xlabel('# patterns')
    plt.ylabel('% convergence (Hamming distance)')
    plt.xticks(n_patterns)
    plt.title('Pictures')
    plt.show()


def capacity_random_patterns():
    patterns = sign(np.random.randn(200, 1024))
    n_patterns = np.arange(1, len(patterns))
    noise_perc = 0.1
    ratios = []

    hn = HopfieldNetwork(self_connections=False)
    for n in n_patterns:
        # learn another pattern (Hebbian learning is incremental!)
        stored_patterns = patterns[0:n]
        hn.learn(stored_patterns[-1].reshape(1, -1))

        # check convergence of distorted patterns
        ratios.append(0)
        for pattern in stored_patterns:
            dist_pattern = u.add_noise(pattern, noise_perc)
            recall = hn.recall(dist_pattern, synchronous=True, max_iters=100)[0]
            ratios[-1] += measure_convergence(pattern, recall)
        ratios[-1] /= n
        print(n)

    plt.plot(n_patterns, ratios)
    plt.xlabel('# patterns')
    plt.ylabel('% convergence (Hamming distance)')
    plt.xticks(np.arange(20, n_patterns[-1], 20))
    plt.title('Random patterns')
    plt.show()


def catastrophic_forgetting():
    n_patterns = np.arange(2, 301)
    dimension = 100
    noise_perc = 0.1
    patterns_ = np.random.randn(n_patterns[-1], dimension)

    for self_connections, bias in zip([False, True, False], [False, False, True]):
        if bias:
            patterns = sign(patterns_ + 0.5)
        else:
            patterns = sign(patterns_)

        hn = HopfieldNetwork(self_connections)
        hn.learn(patterns[0].reshape(1, -1))    # learn first pattern
        counter_pure = []
        counter_dist = []

        for n in n_patterns:
            # learn another pattern (Hebbian learning is incremental!)
            stored_patterns = patterns[0:n]
            hn.learn(stored_patterns[-1].reshape(1, -1))

            counter_pure.append(0)
            counter_dist.append(0)
            for pattern in stored_patterns[:-1]:
                # stability of pure pattern
                recall = hn.recall(pattern, synchronous=True, max_iters=1)[0]
                if np.array_equal(recall, pattern):
                    counter_pure[-1] += 1

                # convergence of distorted pattern
                dist_pattern = u.add_noise(pattern, noise_perc)
                recall = hn.recall(dist_pattern, synchronous=True, max_iters=100)[0]
                if np.array_equal(recall, pattern):
                    counter_dist[-1] += 1
            counter_pure[-1] /= (n-1)
            counter_dist[-1] /= (n-1)
            print(n)

        end_label = ''
        if self_connections:
            end_label += ' with self connections'
        if bias:
            end_label += ' with bias'
        plt.plot(n_patterns, counter_pure, label='stable pure patterns' + end_label)
        plt.plot(n_patterns, counter_dist, label='converged noisy patterns' + end_label)

    plt.xlabel('# patterns')
    plt.ylabel('%')
    plt.xticks(np.arange(20, n_patterns[-1], 20))
    plt.legend()
    plt.show()


np.random.seed(1)
patterns = u.load_pictures()

# assignment3_5_1()
# assignment3_5_2()
catastrophic_forgetting()
