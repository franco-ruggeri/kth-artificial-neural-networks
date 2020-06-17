from util import *
from dbn import DeepBeliefNet


def clear_dir(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        os.remove(file_path)


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

    # results (to get mean and std over multiple runs)
    n_runs = 5
    acc_train_pretrained = np.zeros(n_runs)
    acc_train_finetuned = np.zeros(n_runs)
    acc_test_pretrained = np.zeros(n_runs)
    acc_test_finetuned = np.zeros(n_runs)

    for n in range(n_runs):
        np.random.seed(n)
        print('run {}'.format(n))

        # clear directory (otherwise it loads instead of training)
        clear_dir('trained_rbm')
        clear_dir('trained_dbn')

        dbn = DeepBeliefNet(
            sizes={"vis": image_size[0] * image_size[1], "hid": 500, "pen": 500, "top": 2000, "lbl": n_labels},
            image_size=image_size,
            batch_size=batch_size,
            simplified=True
        )

        # pre-train
        dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=n_iterations)

        # evaluate
        acc_train_pretrained[n] = dbn.recognize(true_img=train_imgs, true_lbl=train_lbls)
        acc_test_pretrained[n] = dbn.recognize(true_img=test_imgs, true_lbl=test_lbls)
        if n == n_runs-1:
            plot_generated_images(dbn=dbn, image_size=image_size, n_labels=n_labels, n_images=10, name='pretrained')

        # fine-tune
        dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=n_iterations)
        if n == n_runs-1:
            plot_accuracy(dbn=dbn, filename='accuracy_dbn.jpg')

        # evaluate
        acc_train_finetuned[n] = dbn.recognize(true_img=train_imgs, true_lbl=train_lbls)
        acc_test_finetuned[n] = dbn.recognize(true_img=test_imgs, true_lbl=test_lbls)
        if n == n_runs-1:
            plot_generated_images(dbn=dbn, image_size=image_size, n_labels=n_labels, n_images=10, name='finetuned')
        print()

    # print results
    print('training accuracy - pre-trained: mean={}, std={}'
          .format(acc_train_pretrained.mean(), acc_train_pretrained.std()))
    print('training accuracy - fine-tuned: mean={}, std={}'
          .format(acc_train_finetuned.mean(), acc_train_finetuned.std()))
    print('test accuracy - pre-trained: mean={}, std={}'
          .format(acc_test_pretrained.mean(), acc_test_pretrained.std()))
    print('test accuracy - fine-tuned: mean={}, std={}'
          .format(acc_test_finetuned.mean(), acc_test_finetuned.std()))
