import numpy as np
import pandas as pd
from .missdata import miss_to_mean

def read_data(file, delimiter, class_split, header = 'infer', index = None, label = None):
    data = pd.read_table(file[0], header = header, index_col = index, delimiter = delimiter)

    if class_split:
        y = pd.read_table(file[1], header = header, delimiter = delimiter, usecols = [0])
        y = y.values.T[0]
        X = data
    else:
        y = data.iloc[:, data.shape[1] - 1]
        y = y.values
        X = data.iloc[:, 0:(data.shape[1] - 2)]

    if X.isnull().values.any():
        X = miss_to_mean(X)

    if label is not None:
        for new_label, old_label in label.items():
            y[y == old_label] = new_label

    return [np.array(X.values, dtype='float64'), y]
