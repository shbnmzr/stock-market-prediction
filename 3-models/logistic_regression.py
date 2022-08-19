import numpy as np
import copy


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def logitic_regression(X, w, b):
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)
    return f_wb


def compute_cost_logistic_regression(X, y, w, b, lambda_=1):
    m, n = X.shape
    f_wb = logitic_regression(X, w, b)
    cost = np.sum(y*np.log(f_wb) + (1 - y) * np.log(1 - f_wb))
    cost /= -m

    cost_reg = (lambda_ / (2 * m)) * np.sum(w ** 2)
    total_cost = cost + cost_reg

    return total_cost


def compute_gradient_logistic_regression(X, y, w, b, lambda_):
    m, n = X.shape
    dj_dw = np.zeros((n, ))
    dj_db = 0.
    f_wb = logitic_regression(X, w, b)
    err = f_wb - y
    for i in range(m):
        for j in range(n):
            dj_dw[j] += err[i] * X[i, j]

    dj_db += np.sum(err)

    for j in range(n):
        dj_dw[j] += lambda_ * w[j]

    dj_db /= m
    dj_dw /= m
    return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, alpha, lambda_, num_iters):
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic_regression(X, y, w, b, lambda_)
        w -= alpha * dj_dw
        b -= dj_db

    return w, b
