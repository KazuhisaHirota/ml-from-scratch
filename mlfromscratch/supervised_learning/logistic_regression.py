import numpy as np
import math

from mlfromscratch.utils.data_manipulation import make_diagonal
from mlfromscratch.deep_learning.activation_functions import Sigmoid


class LogisticRegression():

    def __init__(self, learning_rate=.1, gradient_descent=True):
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.sigmoid = Sigmoid()
        self.param = None

    def _initialize_parameters(self, X):
        n_features = np.shape(X)[1] # [0]: n_samples
        
        limit = 1. / math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features, ))

    def _output(self, X):
        return self.sigmoid(X.dot(self.param)) # \sigma(X.w)

    def fit(self, X, y, n_iterations=4000):
        self._initialize_parameters(X)
        # Tune parameters
        for i in range(n_iterations):
            # Make a new prediction
            y_pred = self._output(X)
            if self.gradient_descent:
                self.param -= self.learning_rate * -(y - y_pred).dot(X)
            else:
                # Make a diagonal matrix of the sigmoid gradient column vector
                diag_gradient = make_diagonal(self.sigmoid.gradient(X.dot(self.param)))
                # Batch optimization:
                self.param = np.linalg.pinv(X.T.dot(diag_gradient).dot(X)) \
                                .dot(X.T).dot(diag_gradient.dot(X).dot(self.param) + y - y_pred)

    def predict(self, X):
        y_pred = np.round(self._output(X)).astype(int) # casted to 0 or 1
        return y_pred

        