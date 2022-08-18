import numpy as np
import pandas as pd
import os


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


def get_correlated_columns(data: pd.DataFrame):
    correlation_matrix = data.corr().abs()
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
    return to_drop.copy()


def drop_correlated_columns(data: pd.DataFrame):
    to_drop = get_correlated_columns(data)
    return data.drop(columns=to_drop)
