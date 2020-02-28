from my_utility import *
import numpy as np
from scipy.special import expit as sigmoid
from scipy.special import softmax


class RBM:
    """Restricted Boltzmann Machine. See Hinton's paper for the details."""

    def __init__(self, n_hidden, learning_rate=0.01, n_epochs=10, mini_batch_size=10, verbose=False):
        self.n_visible = None
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.verbose = verbose
        self.weights = None
        self.biases_v = None
        self.biases_h = np.random.normal(loc=0, scale=0.01, size=self.n_hidden)
        self.reconstruction_loss = None
        self.recognition = False
        self.n_labels = None

    def learn(self, patterns, labels=None):
        """Learn probabilistic representation of data using CD1 (Contrastive Divergence with one step)."""
        if labels is not None:
            self.recognition = True
            self.n_labels = labels.shape[1]
            patterns = np.concatenate((patterns, labels), axis=1)

        self.n_visible = patterns.shape[1]
        self.weights = np.random.normal(loc=0, scale=0.01, size=(self.n_visible, self.n_hidden))
        self.biases_v = np.random.normal(loc=0, scale=0.01, size=self.n_visible)
        self.reconstruction_loss = [self.compute_reconstruction_loss(patterns)]

        mini_batches = self._get_mini_batches(patterns)

        if self.verbose:
            print('Starting learning...')

        for epoch in range(self.n_epochs):
            for mini_batch in mini_batches:
                # alternating Gibbs sampling
                v_0 = mini_batch
                h_0, p_h_0 = self._compute_hidden(v_0)
                v_1, p_v_1 = self._compute_visible(h_0)
                h_1, p_h_1 = self._compute_hidden(p_v_1)    # using probabilities

                # update weights
                positive = v_0.T.dot(h_0)       # using binary states
                negative = p_v_1.T.dot(p_h_1)   # using probabilities
                dw = self.learning_rate / len(mini_batch) * (positive - negative)
                self.weights += dw

                # update biases of visible nodes
                positive = v_0.sum(axis=0)
                negative = p_v_1.sum(axis=0)    # using probabilities
                dw = self.learning_rate / len(mini_batch) * (positive - negative)
                self.biases_v += dw

                # update biases of hidden nodes
                positive = p_h_0.sum(axis=0)    # using probabilities
                negative = p_h_1.sum(axis=0)    # using probabilities
                dw = self.learning_rate / len(mini_batch) * (positive - negative)
                self.biases_h += dw

            # save stats
            self.reconstruction_loss.append(self.compute_reconstruction_loss(patterns))

            if self.verbose:
                print('epoch {} completed - reconstruction loss: {}'.format(epoch, self.reconstruction_loss[-1]))

    def generate(self, n_samples, labels=None, n_gibbs=200):
        """Generate samples."""
        # init visible units sampling from bias
        h = np.zeros((n_samples, self.n_hidden))
        v = self._compute_visible(h)[0]

        # Gibbs sampling to converge
        for i in range(n_gibbs):
            if self.recognition:
                v[:, -self.n_labels:] = labels  # clamp labels to force to generate samples of those classes
            v = self.recall(v)

        if self.recognition:
            v = v[:, :-self.n_labels]
        return v

    def predict(self, patterns, n_gibbs=20):
        """Predict class of samples."""
        if not self.recognition:
            raise RuntimeError('RBM not trained for recognition.')

        labels = 1. / self.n_labels * np.ones((len(patterns), self.n_labels))
        v = np.concatenate((patterns, labels), axis=1)

        # Gibbs sampling to converge
        for i in range(n_gibbs-1):
            v = self.recall(v)

        # TODO: e' ancora buggato, voglio selezionare la label con massima probabilita' invece di campionare!
        h = self._compute_hidden(v)[0]
        v = self._compute_visible(h)[1]
        v = v[:, -self.n_labels:]
        print(np.argmax(v, axis=1))
        v[np.argmax(v, axis=1)] = 1
        v[v < 1] = 0
        print(v)

        # read result
        return v

    def recall(self, patterns, prob=False):
        """Reconstruct samples using one step of alternating Gibbs sampling."""
        hiddens = self._compute_hidden(patterns)[0]
        if prob:    # return probabilities
            reconstructions = self._compute_visible(hiddens)[1]
        else:
            reconstructions = self._compute_visible(hiddens)[0]
        return reconstructions

    def compute_reconstruction_loss(self, patterns):
        """Compute average reconstruction loss."""
        reconstructions = self.recall(patterns)
        return 1. / len(patterns) * np.linalg.norm(reconstructions - patterns)

    def compute_accuracy(self, patterns, labels, n_gibbs=20):
        """Compute the accuracy in predictions."""
        if not self.recognition:
            raise RuntimeError('RBM not trained for recognition.')
        predictions = self.predict(patterns, n_gibbs)
        return len(np.where(np.all(predictions == labels, axis=1))[0]) / len(patterns)

    def _get_mini_batches(self, patterns):
        """Split in mini-batches."""
        n_patterns = len(patterns)
        n_mini_batches = int(np.ceil(float(n_patterns) / self.mini_batch_size))
        mini_batches = []
        for n in range(n_mini_batches):
            start_idx = n * self.mini_batch_size
            end_idx = (n + 1) * self.mini_batch_size
            mini_batches.append(patterns[start_idx:end_idx])
        return mini_batches

    def _compute_hidden(self, visible):
        """Compute probability of activations and sample activations of hidden nodes."""
        p_h = sigmoid(self.biases_h + visible.dot(self.weights))
        h = sample_binary(p_h)
        return h, p_h

    def _compute_visible(self, hidden):
        """Compute probability of activations and sample activations of visible nodes."""
        if self.recognition:
            aux = self.biases_v + hidden.dot(self.weights.T)
            p_v_data = sigmoid(aux[:, :-self.n_labels])
            v_data = sample_binary(p_v_data)
            p_v_labels = softmax(aux[:, -self.n_labels:])
            v_labels = sample_categorical(p_v_labels)
            v = np.concatenate((v_data, v_labels), axis=1)
            p_v = np.concatenate((p_v_data, p_v_labels), axis=1)
        else:
            p_v = sigmoid(self.biases_v + hidden.dot(self.weights.T))
            v = sample_binary(p_v)
        return v, p_v
