from util import *
from rbm import RestrictedBoltzmannMachine


class DeepBeliefNet:
    """
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets.
    https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis] 
                               `-> [lbl] 
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    """
    
    def __init__(self, sizes, image_size, batch_size):
        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          batch_size: Size of mini-batch
        """

        # architecture
        self.rbm_stack = {
            'vis--hid': RestrictedBoltzmannMachine(ndim_visible=sizes['vis'], ndim_hidden=sizes['hid'],
                                                   is_bottom=True, image_size=image_size, batch_size=batch_size),
            'hid--pen': RestrictedBoltzmannMachine(ndim_visible=sizes['hid'], ndim_hidden=sizes['pen'],
                                                   batch_size=batch_size),
            'pen+lbl--top': RestrictedBoltzmannMachine(ndim_visible=sizes['pen']+sizes['lbl'], ndim_hidden=sizes['top'],
                                                       is_top=True, n_labels=sizes['lbl'], batch_size=batch_size)
        }
        self.sizes = sizes
        self.image_size = image_size
        self.n_labels = sizes['lbl']

        # hyper-parameters for usage
        self.n_gibbs_recog = 20
        self.n_gibbs_gener = 200

        # hyper-parameters for learning
        self.batch_size = batch_size
        self.n_gibbs_wakesleep = 5
        self.print_period = 500

    def recognize(self, true_img, true_lbl):
        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_img: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer).
            Used only for calculating accuracy, not driving the net.
        """
        vis = true_img                                      # visible layer gets the image data
        lbl = np.ones(true_lbl.shape)/self.sizes['lbl']     # start the net by telling you know nothing about labels
        
        # [TODO TASK 4.2] fix the image data in the visible layer and drive the network bottom to top. In the top
        # RBM, run alternating Gibbs sampling and read out the labels (replace pass below and 'predicted_lbl' to
        # your predicted labels).
        # NOTE: inferring entire train/test set may require too much compute memory (depends on your system). In
        # that case, divide into mini-batches.

        # single-run sampling bottom to top
        vis = self.rbm_stack['vis--hid'].get_h_given_v_dir(vis)[1]  # binary samples
        vis = self.rbm_stack['hid--pen'].get_h_given_v_dir(vis)[1]

        # add labels units (in the last units, on the right!)
        vis = np.concatenate((vis, lbl), axis=1)

        # alternating Gibbs sampling in top RBM
        for _ in range(self.n_gibbs_recog):
            aux = self.rbm_stack["pen+lbl--top"].reconstruct(vis)
            vis[:, -self.n_labels:] = aux[:, -self.n_labels:]       # representations are clamped for all iterations

        # read results
        predicted_lbl = vis[:, -self.n_labels:]
        accuracy = np.mean(np.argmax(predicted_lbl, axis=1) == np.argmax(true_lbl, axis=1))
        return accuracy

    def generate(self, true_lbl):
        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
        """
        n_samples = true_lbl.shape[0]
        lbl = true_lbl

        # [TODO TASK 4.2] fix the label in the label layer and run alternating Gibbs sampling in the top RBM. From
        #  the top RBM, drive the network top to the bottom visible layer (replace 'vis' from random to your
        #  generated visible layer).

        # initialize penultimate layer with up-pass from random image
        vis = sample_binary(0.5 * np.ones((n_samples, self.sizes['vis'])))
        vis = self.rbm_stack['vis--hid'].get_h_given_v_dir(vis)[1]
        vis = self.rbm_stack['hid--pen'].get_h_given_v_dir(vis)[1]

        # clamp labels (in the last units, on the right!)
        vis = np.concatenate((vis, lbl), axis=1)

        # alternating Gibbs sampling in the top RBM
        for _ in range(self.n_gibbs_gener):
            aux = self.rbm_stack["pen+lbl--top"].reconstruct(vis, sample_v1=True)   # binary samples
            vis[:, :-self.n_labels] = aux[:, :-self.n_labels]       # labels are clamped for all iterations

        # throw away labels units
        vis = vis[:, :-self.n_labels]

        # single-run sampling top to bottom
        vis = self.rbm_stack["hid--pen"].get_v_given_h_dir(vis)[1]  # binary samples
        vis = self.rbm_stack["vis--hid"].get_v_given_h_dir(vis)[0]  # probabilities (images, avoid further noise)

        return vis

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations):
        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the
        entire RBM stack. If not found, learns layer-by-layer (which needs to be completed). Notice that once you
        stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """
        try:
            self.loadfromfile_rbm(loc='trained_rbm', name='vis--hid')
            self.rbm_stack['vis--hid'].untwine_weights()
            
            self.loadfromfile_rbm(loc='trained_rbm', name='hid--pen')
            self.rbm_stack['hid--pen'].untwine_weights()

            self.loadfromfile_rbm(loc='trained_rbm', name='pen+lbl--top')
        except IOError:

            # [TODO TASK 4.2] use CD-1 to train all RBMs greedily
            f, ax = plt.subplots()

            """
            CD-1 training for vis--hid 
            """
            print('training vis--hid')
            vis = vis_trainset
            self.rbm_stack['vis--hid'].cd1(visible_trainset=vis, n_iterations=n_iterations)
            self.savetofile_rbm(loc='trained_rbm', name='vis--hid')
            self.rbm_stack['vis--hid'].untwine_weights()
            plot_reconstruction_loss(rbm=self.rbm_stack['vis--hid'], figure=f, axes=ax, label='vis--hid',
                                     filename='rec_loss_stack.jpg')

            """ 
            CD-1 training for hid--pen 
            """
            print('\ntraining hid--pen')
            vis = self.rbm_stack['vis--hid'].get_h_given_v_dir(vis)[0]      # probabilities
            self.rbm_stack['hid--pen'].cd1(visible_trainset=vis, n_iterations=n_iterations)
            self.savetofile_rbm(loc='trained_rbm', name='hid--pen')
            self.rbm_stack['hid--pen'].untwine_weights()
            plot_reconstruction_loss(rbm=self.rbm_stack['hid--pen'], figure=f, axes=ax, label='hid--pen',
                                     filename='rec_loss_stack.jpg')

            """ 
            CD-1 training for pen+lbl--top 
            """
            print('\ntraining pen+lbl--top')
            vis = self.rbm_stack['hid--pen'].get_h_given_v_dir(vis)[0]      # probabilities
            vis = np.concatenate((vis, lbl_trainset), axis=1)               # add labels (on the right!)
            self.rbm_stack['pen+lbl--top'].cd1(visible_trainset=vis, n_iterations=n_iterations)
            self.savetofile_rbm(loc='trained_rbm', name='pen+lbl--top')

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):
        """
        Wake-sleep method for learning all the parameters of network. 
        First tries to load previous saved parameters of the entire network.

        See appendix B of Hinton et al.'s paper.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """
        print('\ntraining wake-sleep..')

        try:
            self.loadfromfile_dbn(loc='trained_dbn', name='vis--hid')
            self.loadfromfile_dbn(loc='trained_dbn', name='hid--pen')
            self.loadfromfile_rbm(loc='trained_dbn', name='pen+lbl--top')
        except IOError:
            n_samples = vis_trainset.shape[0]
            n_batches = int(n_samples / self.batch_size)

            for it in range(n_iterations):
                # select batch
                batch = it % n_batches
                batch_vis = vis_trainset[batch * self.batch_size:(batch + 1) * self.batch_size]
                batch_lbl = lbl_trainset[batch * self.batch_size:(batch + 1) * self.batch_size]

                # [TODO TASK 4.3] wake-phase: drive the network bottom to top fixing the visible and label data.
                # drive bottom to top (recognition weights)
                wake_hid_p, wake_hid = self.rbm_stack['vis--hid'].get_h_given_v_dir(batch_vis)
                wake_pen_p, wake_pen = self.rbm_stack['hid--pen'].get_h_given_v_dir(wake_hid)

                # positive phase CDk on top RBM
                pos_pen = np.concatenate((wake_pen, batch_lbl), axis=1)
                pos_top_p, pos_top = self.rbm_stack['pen+lbl--top'].get_h_given_v(pos_pen)

                # [TODO TASK 4.3] alternating Gibbs sampling in the top RBM for k='n_gibbs_wakesleep' steps, also store necessary information for learning this RBM.
                # negative phase CDk on top RBM
                neg_top = pos_top      # init loop
                neg_pen = None
                for i in range(self.n_gibbs_wakesleep):
                    neg_pen_p, neg_pen = self.rbm_stack['pen+lbl--top'].get_v_given_h(neg_top)

                    # binary samples for representations, probabilities for labels (softmax)
                    neg_pen = np.concatenate((neg_pen[:, :-self.n_labels], neg_pen_p[:, -self.n_labels:]), axis=1)

                    neg_top_p, neg_top = self.rbm_stack['pen+lbl--top'].get_h_given_v(neg_pen)

                # [TODO TASK 4.3] sleep phase: from the activities in the top RBM, drive the network top to bottom.
                sleep_pen = neg_pen[:, :-self.n_labels]     # throw away label units
                sleep_hid_p, sleep_hid = self.rbm_stack['hid--pen'].get_v_given_h_dir(sleep_pen)
                sleep_vis_p, sleep_vis = self.rbm_stack['vis--hid'].get_v_given_h_dir(sleep_hid)

                # [TODO TASK 4.3] compute predictions: compute generative predictions from wake-phase activations, and recognize predictions from sleep-phase activations.
                # predictions during wake phase (for generative weights)
                pred_wake_vis = self.rbm_stack['vis--hid'].get_v_given_h_dir(wake_hid)[0]
                pred_wake_hid = self.rbm_stack['hid--pen'].get_v_given_h_dir(wake_pen)[0]

                # predictions during sleep phase (for recognition weights)
                pred_sleep_pen = self.rbm_stack['hid--pen'].get_h_given_v_dir(sleep_hid)[0]
                pred_sleep_hid = self.rbm_stack['vis--hid'].get_h_given_v_dir(sleep_vis_p)[0]

                # [TODO TASK 4.3] update generative parameters: here you will only use 'update_generate_params' method from rbm class.
                self.rbm_stack['vis--hid'].update_generate_params(inps=wake_hid, trgs=batch_vis, preds=pred_wake_vis)
                self.rbm_stack['hid--pen'].update_generate_params(inps=wake_pen, trgs=wake_hid, preds=pred_wake_hid)

                # [TODO TASK 4.3] update parameters of top rbm: here you will only use 'update_params' method from rbm class.
                self.rbm_stack['pen+lbl--top'].update_params(v_0=pos_pen, h_0=pos_top, v_k=neg_pen, h_k=neg_top)

                # [TODO TASK 4.3] update generative parameters: here you will only use 'update_recognize_params' method from rbm class.
                self.rbm_stack['hid--pen'].update_recognize_params(inps=sleep_hid, trgs=sleep_pen, preds=pred_sleep_pen)
                self.rbm_stack['vis--hid'].update_recognize_params(inps=sleep_vis_p, trgs=sleep_hid, preds=pred_sleep_hid)

                if it % self.print_period == 0:
                    print('iteration=%07d' % it)

            self.savetofile_dbn(loc="trained_dbn", name="vis--hid")
            self.savetofile_dbn(loc="trained_dbn", name="hid--pen")
            self.savetofile_rbm(loc="trained_dbn", name="pen+lbl--top")

    def loadfromfile_rbm(self, loc, name):
        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy" % (loc, name))
        self.rbm_stack[name].bias_v = np.load("%s/rbm.%s.bias_v.npy" % (loc, name))
        self.rbm_stack[name].bias_h = np.load("%s/rbm.%s.bias_h.npy" % (loc, name))
        print("loaded rbm[%s] from %s" % (name, loc))
        
    def savetofile_rbm(self, loc, name):
        np.save("%s/rbm.%s.weight_vh" % (loc, name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v" % (loc, name), self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h" % (loc, name), self.rbm_stack[name].bias_h)
    
    def loadfromfile_dbn(self, loc, name):
        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy" % (loc, name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy" % (loc, name))
        self.rbm_stack[name].bias_v = np.load("%s/dbn.%s.bias_v.npy" % (loc, name))
        self.rbm_stack[name].bias_h = np.load("%s/dbn.%s.bias_h.npy" % (loc, name))
        print("loaded rbm[%s] from %s" % (name, loc))
        
    def savetofile_dbn(self, loc, name):
        np.save("%s/dbn.%s.weight_v_to_h" % (loc, name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v" % (loc, name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v" % (loc, name), self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h" % (loc, name), self.rbm_stack[name].bias_h)
