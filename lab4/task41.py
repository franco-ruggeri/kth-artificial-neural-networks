from util import *
from rbm import RestrictedBoltzmannMachine


if __name__ == '__main__':
    # load data
    image_size = (28, 28)
    n_labels = 10
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)
    plot_histogram_classes(train_labels=train_lbls, test_labels=test_lbls, n_labels=n_labels)
    plot_images(images=train_imgs[:16], image_size=image_size, grid=(4, 4), filename='train_images.jpg')

    # hyper-parameters
    batch_size = 20
    n_epochs = 10
    n_iterations = n_epochs * int(train_imgs.shape[0] / batch_size)
    ndim_hidden = [200, 300, 500]

    # results (to get mean and std over multiple runs)
    n_runs = 5
    rec_loss_train = np.zeros((len(ndim_hidden), n_runs))
    rec_loss_test = np.zeros((len(ndim_hidden), n_runs))

    for n in range(n_runs):
        np.random.seed(n)
        print('run {}'.format(n))

        if n == n_runs-1:
            f, ax = plt.subplots()

        for i in range(len(ndim_hidden)):
            # train
            rbm = RestrictedBoltzmannMachine(
                ndim_visible=image_size[0] * image_size[1],
                ndim_hidden=ndim_hidden[i],
                is_bottom=True,
                image_size=image_size,
                is_top=False,
                n_labels=n_labels,
                batch_size=batch_size
            )
            rbm.cd1(visible_trainset=train_imgs, n_iterations=n_iterations)

            # evaluate
            rec_loss_train[i, n] = rbm.compute_reconstruction_loss(train_imgs)
            rec_loss_test[i, n] = rbm.compute_reconstruction_loss(test_imgs)

            if n == n_runs-1:
                # plot reconstruction loss
                plot_reconstruction_loss(rbm=rbm, figure=f, axes=ax, label='{} units'.format(ndim_hidden[i]),
                                         filename='rec_loss_rbm.jpg')

                # plot some reconstructions (from test set)
                plot_reconstructions(rbm=rbm, images=test_imgs, image_size=image_size,
                                     filename='test_reconstructions_{}'.format(ndim_hidden[i]))
        print()

    # print results
    for i in range(len(ndim_hidden)):
        print('training reconstruction loss {} nodes: mean={}, std={}'
              .format(ndim_hidden[i], rec_loss_train[i].mean(), rec_loss_train[i].std()))
    for i in range(len(ndim_hidden)):
        print('test reconstruction loss {} nodes: mean={}, std={}'
              .format(ndim_hidden[i], rec_loss_test[i].mean(), rec_loss_test[i].std()))
