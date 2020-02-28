from my_utility import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
from my_rbm import RBM


def load_or_train(filename, n_hidden, mini_batch_size, n_epochs, recognition=False):
    try:
        with open(filename, 'rb') as f:
            rbm = pickle.load(f)
    except IOError:
        rbm = RBM(n_hidden=n_hidden, mini_batch_size=mini_batch_size, n_epochs=n_epochs, verbose=True)
        if recognition:
            rbm.learn(train_images, train_labels)
        else:
            rbm.learn(train_images)
        with open(filename, 'wb') as f:
            pickle.dump(rbm, f)
    return rbm


np.random.seed(1)

# load data
train_images, train_labels, test_images, test_labels = read_mnist()
image_size = (28, 28)

# compare reconstruction losses for different number of hidden units
for n_hidden in [50, 100, 200, 500]:
    rbm = load_or_train('trained_rbm/rbm{}'.format(n_hidden), n_hidden=n_hidden, mini_batch_size=20, n_epochs=10)

    plt.plot(rbm.reconstruction_loss, label='n_hidden={}'.format(n_hidden))
    plt.title('Reconstruction loss')
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.xticks(np.arange(rbm.n_epochs + 1))
plt.legend()
plt.show()

# reconstruction of some images
rbm = load_or_train('trained_rbm/rbm500', n_hidden=500, mini_batch_size=20, n_epochs=10)
n_images = 4
idx = np.random.choice(len(train_images), size=n_images)
for i, image in enumerate(train_images[idx]):
    plt.subplot(n_images, 2, i*2+1)
    plt.imshow(image.reshape(image_size))
    plt.xticks([])
    plt.yticks([])
    plt.subplot(n_images, 2, i*2+2)
    plt.imshow(rbm.recall(image).reshape(image_size))
    plt.xticks([])
    plt.yticks([])
plt.show()

# generate samples
rbm = load_or_train('trained_rbm/rbm500', n_hidden=500, mini_batch_size=20, n_epochs=10)
grid = (4, 4)
n_images = grid[0] * grid[1]
for i, image in enumerate(rbm.generate(n_images)):
    plt.subplot(grid[0], grid[1], i+1)
    plt.imshow(image.reshape(image_size))
    plt.xticks([])
    plt.yticks([])
plt.show()

# recognition
rbm = load_or_train('trained_rbm/rbm_recognition500', n_hidden=500, mini_batch_size=20, n_epochs=10, recognition=True)
# print('Accuracy on training set:', rbm.compute_accuracy(train_images, train_labels))
print('Accuracy on test set:', rbm.compute_accuracy(test_images, test_labels))
