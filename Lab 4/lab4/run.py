from util import *
from rbm import RestrictedBoltzmannMachine
from dbn import DeepBeliefNet

if __name__ == "__main__":
    np.random.seed(100)

    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    ''' Restricted Boltzmann Machine '''
    print ("\nStarting a Restricted Boltzmann Machine..")

    # training
    batch_size = 20
    n_epochs = 10
    plot_data = []
    for ndim_hidden in [50, 100, 200, 500]:
        rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0] * image_size[1],
                                         ndim_hidden=ndim_hidden,
                                         is_bottom=True,
                                         image_size=image_size,
                                         is_top=False,
                                         n_labels=10,
                                         batch_size=batch_size)
        rbm.cd1(visible_trainset=train_imgs, n_epochs=n_epochs)
        plot_data.append((ndim_hidden, rbm.reconstruction_error))

    # show trend of reconstruction error
    for ndim_hidden, reconstruction_error in plot_data:
        plt.plot(reconstruction_error, label='n_hidden={}'.format(ndim_hidden))
    plt.xticks(np.arange(n_epochs+1))
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('reconstruction error')
    plt.savefig('images/reconstruction_error.png')
    plt.close()

    # show some reconstructions
    n_images = 2
    for i, n in enumerate(np.random.choice(len(train_imgs), replace=False, size=n_images)):
        plt.subplot(n_images, 2, i*2+1)
        plt.imshow(train_imgs[n].reshape(image_size))
        plt.subplot(n_images, 2, i*2+2)
        plt.imshow(rbm.gibbs(train_imgs[n]).reshape(image_size))
    plt.savefig('images/reconstructions.png')
    plt.close()


    ''' Deep-Belief Net '''
    print ("\nStarting a Deep Belief Net..")
    dbn = DeepBeliefNet(sizes={"vis": image_size[0] * image_size[1], "hid": 500, "pen": 500, "top": 2000, "lbl": 10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=10
                        )

    ''' Greedy layer-wise training '''
    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_epochs=1)
    dbn.recognize(train_imgs, train_lbls)
    dbn.recognize(test_imgs, test_lbls)
    for digit in range(10):
        digit_1hot = np.zeros(shape=(1, 10))
        digit_1hot[0, digit] = 1
        dbn.generate(digit_1hot, name="rbms")

    ''' Fine-tune wake-sleep training '''
    dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=2000)
    dbn.recognize(train_imgs, train_lbls)
    dbn.recognize(test_imgs, test_lbls)
    for digit in range(10):
        digit_1hot = np.zeros(shape=(1, 10))
        digit_1hot[0, digit] = 1
        dbn.generate(digit_1hot, name="dbn")
