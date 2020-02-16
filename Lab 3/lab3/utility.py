import matplotlib.pyplot as plt


def check_stability(hn, patterns):
    for pattern in patterns:
        result = hn.recall(pattern, synchronous=True, max_iters=1)
        if not result[2] or any(result[0] != pattern):
            return False
    return True


def plot_image(pattern, title=None):
    pattern = pattern.reshape(32, 32)
    plt.imshow(pattern)
    if title is not None:
        plt.title(title)
    plt.show()


def plot_energy(energy, title=None):
    plt.plot(energy)
    if title is not None:
        plt.title(title)
    plt.xlabel('iteration')
    plt.ylabel('energy')
    plt.xlim([0, len(energy)])
    plt.show()
