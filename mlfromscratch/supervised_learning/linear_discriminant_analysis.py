import numpy as np

from mlfromscratch.utils.data_operation import calculate_covariance_matrix


class LDA():

    def __init__(self):
        self.w = None

    def transform(self, X, y):
        self.fit(X, y)
        # Project data onto vector
        X_transform = X.dot(self.w)
        return X_transform

    def fit(self, X, y):
        # Separate data by class
        X1 = X[y == 0]
        X2 = X[y == 1]

        cov1 = calculate_covariance_matrix(X1)
        cov2 = calculate_covariance_matrix(X2)
        cov_tot = cov1 + cov2

        mean1 = X1.mean(0) # direction 0 is sample direction
        mean2 = X2.mean(0) # direction 0 is sample direction
        mean_diff = np.atleast_1d(mean1 - mean2) # vectorize the scalar

        # w = (mean1 - mean2) / (cov1 + cov2)
        self.w = np.linalg.pinv(cov_tot).dot(mean_diff)

    def predict(self, X):
        y_pred = []
        for sample in X: # pick up an row (size: (1, n_features))
            h = sample.dot(self.w) # w (size: (n_features, 1))
            y = 1 * (h < 0) # ??
            y_pred.append(y)
        return y_pred
