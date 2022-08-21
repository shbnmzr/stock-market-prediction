import numpy as np


def split_train_dev_test(X, y, train_size=0.7, test_size=0.15):
    train_len = int(len(X) * train_size)
    test_len = int(len(X) * test_size)
    dev_len = len(X) - (train_len + test_len)
    X_train, y_train, X_dev, y_dev, X_test, y_test = np.array(X[:train_len]), np.array(y[:train_len]), np.array(
        X[train_len:(train_len + dev_len)]), np.array(y[train_len:(train_len + dev_len)]), np.array(
        X[(train_len + dev_len):]), np.array(y[(train_len + dev_len):])

    return X_train, y_train, X_dev, y_dev, X_test, y_test

def check_balance(X, y):
    pass
