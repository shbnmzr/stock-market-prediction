import numpy as np
import pandas as pd


def drop_correlated_columns(data: pd.DataFrame):
    correlation_matrix = data.corr().abs()
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
    return data.drop(columns=to_drop)
