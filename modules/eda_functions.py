import numpy as np
import pandas as pd


def get_correlated_columns(data: pd.DataFrame):
    correlation_matrix = data.corr().abs()
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
    return to_drop.copy()


def drop_correlated_columns(data: pd.DataFrame):
    to_drop = get_correlated_columns(data)
    return data.drop(columns=to_drop)


def create_dataframe_from_dict(data, column_name):
    df = pd.DataFrame.from_dict(data=data, orient='index', columns=[column_name])
    return df
