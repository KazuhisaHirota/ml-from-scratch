import numpy as np
import math


class Regression(object):

    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        
        self.training_errors = []
        self.w = None # weights of features
        self.regularization = None # function

    def _initialize_weights(self, n_features):
        limit = 1. / math.sqrt(n_features) # NOTE sqrt
        self.w = np.random.uniform(-limit, limit, size=(n_features, )) # NOTE ", )"

    def _output(self, X):
        return X.dot(self.w) # (n_samples, n_features) x n_features

    # default implementation
    def fit(self, X, y):
        X = np.insert(X, 0, 1., axis=1) # 0: insertion index, axis=1: insert ones vertically
        self._initialize_weights(n_features=X.shape[1]) # .shape[0]: n_samples

        # Do gradient descent for n_iterations
        for i in range(self.n_iterations):
            y_pred = self._output(X)
            
            # Calculate L2 loss
            mse = np.mean(0.5 * (y - y_pred)**2 + self.regularization(self.w))
            self.training_errors.append(mse)

            # Gradient of L2 loss w.r.t w
            grad_w = -(y - y_pred).dot(X) + self.regularization.grad(self.w)

            # Update the weights
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        X = np.insert(X, 0, 1., axis=1)
        y_pred = self._output(X)
        return y_pred


class LinearRegression(Regression):

    def __init__(self, n_iterations=100, learning_rate=0.001, gradient_descent=True):
        # at first, initialize the super class
        super().__init__(n_iterations, learning_rate)

        self.gradient_descent = gradient_descent
        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0

    def fit(self, X, y):
        if self.gradient_descent:
            super().fit(X, y)
        else:
            X = np.insert(X, 0, 1., axis=1)
            # Calculate weights by least squares using pseudo-inverse
            U, S, V = np.linalg.svd(X.T.dot(X))
            S = np.diag(S)
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T) # (X^T X)^(-1)
            self.w = X_sq_reg_inv.dot(X.T).dot(y) # (X^T X)^(-1) X^T y