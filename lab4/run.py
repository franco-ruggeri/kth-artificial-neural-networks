from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet


def test_generative_mode(dbn, n_labels, n_images, name):
    for digit in range(n_labels):
        # generate
        digit_1hot = np.zeros(shape=(n_images, n_labels))
        digit_1hot[0, digit] = 1
        gen_imgs = dbn.generate(digit_1hot)

        # stitch video
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([])
        ax.set_yticks([])
        records = []
        for img in gen_imgs:
            records.append(ax.imshow(img.reshape(image_size), cmap="bwr", vmin=0, vmax=1, animated=True,
                                     interpolation=None))
        stitch_video(fig, records, filename='%s.generate%d.mp4'.format(name, digit))


if __name__ == "__main__":
    np.random.seed(1)

    """ Load and examine data """
    image_size = [28, 28]
    n_labels = 10
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)
    plot_histogram_classes(train_labels=train_lbls, test_labels=test_lbls, n_labels=n_labels)
    plot_images(images=train_imgs[:16], image_size=image_size, grid=(4, 4), filename='train_images.jpg')

    """ Restricted Boltzmann Machine """
    print('Starting a Restricted Boltzmann Machine...')

    batch_size = 20
    n_epochs = 10
    n_iterations = n_epochs * int(train_imgs.shape[0] / batch_size)

    f, ax = plt.subplots()
    for n_hidden in [200, 500]:
        rbm = RestrictedBoltzmannMachine(
            ndim_visible=image_size[0] * image_size[1],
            ndim_hidden=n_hidden,
            is_bottom=True,
            image_size=image_size,
            is_top=False,
            n_labels=n_labels,
            batch_size=batch_size
        )

        # train
        rbm.cd1(visible_trainset=train_imgs, n_iterations=n_iterations)
        plot_reconstruction_loss(rbm=rbm, figure=f, axes=ax, label='{} units'.format(n_hidden),
                                 filename='rec_loss_200_vs_500.jpg')

        # evaluate
        print('Test reconstruction loss with {} hidden units: {:4.4f}\n'
              .format(n_hidden, rbm.compute_reconstruction_loss(train_imgs)))

        # show some reconstructions (from test set)
        test_imgs_ = test_imgs[np.random.choice(len(test_imgs), replace=False, size=8)]
        reconstructions = rbm.reconstruct(test_imgs_)
        images = np.zeros((16, image_size[0] * image_size[1]))
        for i in range(4):
            for j in range(4):
                idx = (i // 2) * 4 + j
                if i % 2 == 0:  # even rows => original images
                    images[i*4 + j] = test_imgs_[idx]
                else:           # odd rows => reconstructions
                    images[i * 4 + j] = reconstructions[idx]
        plot_images(images, image_size=image_size, grid=(4, 4), filename='test_reconstructions_{}'.format(n_hidden))
    print()

    """ Deep Belief Net """
    print('Starting a Deep Belief Net...\n')

    dbn = DeepBeliefNet(
        sizes={"vis": image_size[0]*image_size[1], "hid": 500, "pen": 500, "top": 2000, "lbl": n_labels},
        image_size=image_size,
        batch_size=10
    )

    # pre-train
    batch_size = 20
    n_epochs = 10
    n_iterations = n_epochs * int(train_imgs.shape[0] / batch_size)
    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=n_iterations)

    # recognition mode
    print('training accuracy = %.2f%%' % (100. * dbn.recognize(train_imgs, train_lbls)))
    print('test accuracy = %.2f%%' % (100. * dbn.recognize(test_imgs, test_lbls)))

    # generative mode
    test_generative_mode(dbn, n_labels, 10, 'pretrained')

    # fine-tune
    batch_size = 20
    n_epochs = 10
    n_iterations = n_epochs * int(train_imgs.shape[0] / batch_size)
    dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=n_iterations)

    # recognition mode
    print('training accuracy = %.2f%%' % (100. * dbn.recognize(train_imgs, train_lbls)))
    print('test accuracy = %.2f%%' % (100. * dbn.recognize(test_imgs, test_lbls)))

    # generative mode
    test_generative_mode(dbn, n_labels, 10, 'finetuned')
