import numpy as np
import matplotlib.pyplot as plt
import os

dataset_dir = 'dataset'
figures_dir = 'figures'


def sigmoid(support):
    """ 
    Sigmoid activation function that finds probabilities to turn ON each unit. 
        
    Args:
      support: shape is (size of mini-batch, size of layer)      
    Returns:
      on_probabilities: shape is (size of mini-batch, size of layer)      
    """
    on_probabilities = 1./(1.+np.exp(-support))
    return on_probabilities


def softmax(support):
    """ 
    Softmax activation function that finds probabilities of each category
        
    Args:
      support: shape is (size of mini-batch, number of categories)      
    Returns:
      probabilities: shape is (size of mini-batch, number of categories)      
    """
    expsup = np.exp(support-np.sum(support, axis=1)[:, None])
    return expsup / np.sum(expsup, axis=1)[:, None]


def sample_binary(on_probabilities):
    """ 
    Sample activations ON=1 (OFF=0) from probabilities sigmoid probabilities
        
    Args:
      on_probabilities: shape is (size of mini-batch, size of layer)
    Returns:
      activations: shape is (size of mini-batch, size of layer)      
    """
    activations = 1. * (on_probabilities >= np.random.random_sample(size=on_probabilities.shape))
    return activations


def sample_categorical(probabilities):
    """ 
    Sample one-hot activations from categorical probabilities
        
    Args:
      probabilities: shape is (size of mini-batch, number of categories)
    Returns:
      activations: shape is (size of mini-batch, number of categories)      
    """
    cumsum = np.cumsum(probabilities, axis=1)
    rand = np.random.random_sample(size=probabilities.shape[0])[:, None]
    activations = np.zeros(probabilities.shape)
    activations[range(probabilities.shape[0]), np.argmax((cumsum >= rand), axis=1)] = 1
    return activations


def load_idxfile(filename):
    """
    Load idx file format. For more information: http://yann.lecun.com/exdb/mnist/
    """
    import struct
        
    with open(filename, 'rb') as _file:
        if ord(_file.read(1)) != 0 or ord(_file.read(1)) != 0:
            raise Exception('Invalid idx file: unexpected magic number!')
        dtype, ndim = ord(_file.read(1)), ord(_file.read(1))
        shape = [struct.unpack(">I", _file.read(4))[0] for _ in range(ndim)]
        data = np.fromfile(_file, dtype=np.dtype(np.uint8).newbyteorder('>')).reshape(shape)
    return data


def read_mnist(dim=(28, 28), n_train=60000, n_test=1000):
    """
    Read mnist train and test data. Images are normalized to be in range [0,1]. Labels are one-hot coded.
    """
    train_imgs = load_idxfile(os.path.join(dataset_dir, "train-images-idx3-ubyte"))
    train_imgs = train_imgs / 255.
    train_imgs = train_imgs.reshape(-1, dim[0]*dim[1])

    train_lbls = load_idxfile(os.path.join(dataset_dir, "train-labels-idx1-ubyte"))
    train_lbls_1hot = np.zeros((len(train_lbls), 10), dtype=np.float32)
    train_lbls_1hot[range(len(train_lbls)), train_lbls] = 1.

    test_imgs = load_idxfile(os.path.join(dataset_dir, "t10k-images-idx3-ubyte"))
    test_imgs = test_imgs / 255.
    test_imgs = test_imgs.reshape(-1, dim[0]*dim[1])

    test_lbls = load_idxfile(os.path.join(dataset_dir, "t10k-labels-idx1-ubyte"))
    test_lbls_1hot = np.zeros((len(test_lbls), 10), dtype=np.float32)
    test_lbls_1hot[range(len(test_lbls)), test_lbls] = 1.

    return train_imgs[:n_train], train_lbls_1hot[:n_train], test_imgs[:n_test], test_lbls_1hot[:n_test]


def viz_rf(weights, it, grid):
    """
    Visualize receptive fields and save 
    """
    fig, axs = plt.subplots(grid[0], grid[1], figsize=(grid[1], grid[0]))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    imax = abs(weights).max()
    for x in range(grid[0]):
        for y in range(grid[1]):
            axs[x, y].set_xticks([])
            axs[x, y].set_yticks([])
            axs[x, y].imshow(weights[:, :, y+grid[1]*x], cmap="bwr", vmin=-imax, vmax=imax, interpolation=None)
    plt.savefig(os.path.join(figures_dir, "rf.iter%06d.png" % it))
    plt.close('all')


def stitch_video(fig, imgs, filename):
    """
    Stitches a list of images and returns a animation object
    """
    import matplotlib.animation as animation
    
    anim = animation.ArtistAnimation(fig, imgs, interval=500, blit=True, repeat=False)
    anim.save(os.path.join(figures_dir, filename))


# additional functions, by fruggeri

def plot_histogram_classes(train_labels, test_labels, n_labels):
    labels = np.arange(n_labels)
    n_train = train_labels.sum(axis=0)
    n_test = test_labels.sum(axis=0)

    plt.figure()
    plt.bar(labels, n_train, width=.25, label='train')
    plt.bar(labels+.25, n_test, width=.25, label='test')
    plt.legend()
    plt.savefig(os.path.join(figures_dir, 'dataset.png'))
    plt.close()


def plot_images(images, image_size, grid, filename):
    axes = plt.subplots(grid[0], grid[1], figsize=(grid[1], grid[0]))[1]
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    for x in range(grid[0]):
        for y in range(grid[1]):
            idx = x*grid[1]+y
            ax = axes[x, y]
            ax.imshow(images[idx].reshape(image_size), cmap="bwr", vmin=0, vmax=1, interpolation=None)
            ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.savefig(os.path.join(figures_dir, filename))
    plt.close()


def plot_reconstruction_loss(rbm, figure, axes, label, filename):
    axes.plot(rbm.measured_iterations, rbm.reconstruction_loss, label=label)
    axes.set_xlabel('iteration')
    axes.set_ylabel('reconstruction loss')
    axes.legend()
    figure.savefig(os.path.join(figures_dir, filename))
    plt.close(figure)


def plot_reconstructions(rbm, images, image_size, filename):
    images = images[np.random.choice(len(images), replace=False, size=8)]
    reconstructions = rbm.reconstruct(images)
    images_plot = np.zeros((16, image_size[0] * image_size[1]))
    for i in range(4):
        for j in range(4):
            idx = (i // 2) * 4 + j
            if i % 2 == 0:      # even rows => original images
                images_plot[i * 4 + j] = images[idx]
            else:               # odd rows => reconstructions
                images_plot[i * 4 + j] = reconstructions[idx]
    plot_images(images=images_plot, image_size=image_size, grid=(4, 4), filename=filename)


def plot_generated_images(dbn, image_size, n_labels, n_images, name):
    gen_imgs = np.zeros((n_labels, n_images, image_size[0] * image_size[1]))

    for digit in range(n_labels):
        # generate
        digit_1hot = np.zeros(shape=(n_images, n_labels))
        digit_1hot[:, digit] = 1
        gen_imgs[digit] = dbn.generate(digit_1hot)

        # stitch video
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([])
        ax.set_yticks([])
        records = []
        for img in gen_imgs[digit]:
            records.append([ax.imshow(img.reshape(image_size), cmap="bwr", vmin=0, vmax=1, animated=True,
                                      interpolation=None)])
        stitch_video(fig, records, filename='{}.generate{}.mp4'.format(name, digit))
        plt.close('all')

    # plot in figure
    gen_imgs = gen_imgs.reshape(-1, image_size[0] * image_size[1])
    plot_images(images=gen_imgs, image_size=image_size, grid=(n_labels, n_images),
                filename='{}.generate.jpg'.format(name))


def plot_accuracy(dbn, filename):
    plt.figure()
    plt.plot(dbn.measured_iterations, dbn.accuracy)
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.savefig(os.path.join(figures_dir, filename))
    plt.close()
