import numpy as np
import pandas as pd
import subprocess as sub
from sklearn.model_selection import StratifiedKFold

def nodes_split_data(X, y, nodes):
    node = 0
    skf  = StratifiedKFold(n_splits = nodes)
    for splited_index in skf.split(X, y):
        new_X = pd.DataFrame(X[splited_index[1]])
        new_y = pd.DataFrame(y[splited_index[1]])

        X_file = "datasfornode/data_" + str(node) + ".csv"
        y_file = "datasfornode/class_" + str(node) + ".csv"
        new_X.to_csv(X_file, index = False)
        new_y.to_csv(y_file, index = False)
        node += 1


