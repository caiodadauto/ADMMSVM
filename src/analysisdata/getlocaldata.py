import numpy as np
import pandas as pd

def get_local_data():
    X_local_train = pd.read_csv('datasfornode/data_0.csv').values
    y_local_train = pd.read_csv('datasfornode/class_0.csv').values.T[0]

    return [X_local_train, y_local_train]

