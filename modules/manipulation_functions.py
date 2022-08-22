import numpy as np
import pandas as pd
import os


def split_train_dev_test(X, y, train_size=0.7, test_size=0.15):
    train_len = int(len(X) * train_size)
    test_len = int(len(X) * test_size)
    dev_len = len(X) - (train_len + test_len)
    X_train, y_train, X_dev, y_dev, X_test, y_test = np.array(X[:train_len]), np.array(y[:train_len]), np.array(
        X[train_len:(train_len + dev_len)]), np.array(y[train_len:(train_len + dev_len)]), np.array(
        X[(train_len + dev_len):]), np.array(y[(train_len + dev_len):])

    return X_train, y_train, X_dev, y_dev, X_test, y_test


def split_train_test(X, y, train_size=0.85):
    train_len = int(len(X) * train_size)
    X_train, y_train, X_test, y_test = np.array(X[:train_len]), np.array(y[:train_len]), np.array(
        X[train_len:]), np.array(y[train_len:])

    return X_train, y_train, X_test, y_test


def export_predictions(predictions, path):
    try:
        os.mkdir(path)
    except:
        print('Already Exists')
    finally:
        path += '/'
        for ticker in predictions:
            indices = [i+1 for i in range(len(predictions[ticker]['True Values']))]
            indices = pd.Series(indices)
            df = pd.DataFrame(predictions[ticker], index=indices, columns=['True Values', 'Predictions'])
            df.to_csv(path + ticker + '.csv')

    return


def import_all_files_as_dict(path):
    companies = {}
    for file in os.listdir(path):
        if file.endswith('.csv'):
            name = file.split('.')[0]
            companies[name] = pd.DataFrame(pd.read_csv(os.path.join(path, file), index_col='Date', parse_dates=True, skip_blank_lines=True))
            companies[name] = companies[name].dropna(axis=0)
            if len(companies[name]) == 0:
                del companies[name]

    return companies


def import_all_files(path):
    companies = import_all_files_as_dict(path)
    data = pd.concat(companies, keys=companies.keys())
    return data


def import_single_file(path, symbol):
    data = pd.read_csv(os.path.join(path, symbol + '.csv'), index_col='Date', parse_dates=True)
    return data
