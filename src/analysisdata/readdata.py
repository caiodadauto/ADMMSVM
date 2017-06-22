import numpy as np
import pandas as pd
from .missdata import miss_to_mean

def read_data(file, delimiter, class_split, header = 'infer', index = None):
    data = pd.read_table(file[0], header = header, index_col = index, delimiter = delimiter)

    if class_split:
        y = pd.read_table(file[1], header = header, delimiter = delimiter, usecols = [0])
        X = data
    else:
        y = data.iloc[:, data.shape[1]]
        X = data.iloc[:, 0:(data.shape[1] - 1)]

    if X.isnull().values.any():
        X = miss_to_mean(X)

    return [X.values, y.values.T[0]]
