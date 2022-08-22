import pandas as pd
import numpy as np


def create_lag(data, column, lag_number=1):
    lag = data[column].shift(lag_number)
    data[f'lag_{lag_number}'] = lag
    data = data.dropna(axis=0)
    return data


def create_target(data, column):
    price = data[column].shift(-1)
    data['Target'] = price
    data = data.dropna(axis=0)
    return data


def z_score_normalization(X):
    me = np.mean(X)
    sigma = np.std(X)

    normal = (X - me) / sigma
    return normal


def min_max_normaliztion(X):
    min_val = X.min()
    max_val = X.max()
    normal = (X - min_val) / (max_val - min_val)
    return normal
