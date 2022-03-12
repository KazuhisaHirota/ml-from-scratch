import numpy as np

def shuffle_data(X, y, seed=None):
    if seed:
        np.random.seed(seed)
    
    indices = np.arange(X.shape[0]) # row direction: n_samples
    np.random.shuffle(indices)
    return X[indices], y[indices]

def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    
    # train data: the former part,
    # test data: the latter part.
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]
    return X_train, X_test, y_train, y_test