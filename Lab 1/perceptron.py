import numpy as np



class PCT:
    """Single layered perceptron using the delta rule"""

    def __init__(self, data, eta = 0.001, epochs = 20):
        """Initializes the variables"""

        self.data = data
        self.eta = eta
        self.epochs = epochs
        self.w = self.init_weights(data)


    def init_weights(self, inputs):
        """
        :param inputs: Pattern data. Type: Numpy array. Assume dimension is of the form: (nPoints, dimOfData).
        :return: Returns the initialized weight matrix, with elements drawn from a Gaussian distribution.
        """
        mu, sigma = 0, 0.1
        dimension = inputs.shape[0]+1 #The plus one from the bias term, which is added to the input data
        weight_matrix = np.random.normal(mu,sigma,dimension)

        return weight_matrix

    def shape_inputs(self,inputs):
        """This method adds the bias row to the pattern matrix"""
        bias_vector = np.ones(np.shape(inputs)[1])
        return np.vstack([inputs, bias_vector])


    def activation(self, inputs):
        """
        :param inputs: The input patterns.
        :return:  matrix multiplication W*X
        """

        patterns = self.shape_inputs(inputs)
        activations = np.dot(self.w, patterns)
        activations = np.where(activations > 0, 1, -1)

        return activations


    def delta_rule(self, inputs, targets):
        """
        :param X: Input matrix. Type: numpy array. Dimensions: (nPoints, dimOfData)
        :param T: Output matrix. Type: numpy array. Dimensions: (1, nPoints)
        :param W: Weight matrix. Type: numpy array. Dimensions: (1, dimOfData+1)
        :param epochs: Number of learning epochs, this is how many times the weight matrix will be updated.

        :return: Iteratively updated weight matrix, according to number of epochs.
        """

        X = self.shape_inputs(inputs)

        for epoch in range(self.epochs):

            self.activations = self.activation(inputs)
            self.w -= self.eta*np.dot(X, self.activations - targets)

    def show_weights(self):

        return self.w



