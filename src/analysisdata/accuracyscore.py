import numpy as np
import pandas as pd

def accuracy_score(X, y):
    guess_true = 0

    v = pd.read_csv('SVM-ADMM-result.csv').values.T[0]

    n = v.shape[0] - 1
    w = v[0:n]
    b = v[n]

    n_data = X.shape[0]
    for data in range(n_data):
        f = y[data] * (np.dot(w, X[data]) + b)
        if f > 0:
            guess_true += 1

    return guess_true/n_data

