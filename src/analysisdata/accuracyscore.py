import numpy as np
import pandas as pd
from pathlib import Path

def accuracy_score(X, y):
    acc_dict = {}
    pathlist = Path('partialresults/').glob('*.csv')
    for path in pathlist:
        guess_true = 0
        file       = str(path)
        iter       = int(file.rsplit(sep = '_')[-1].rsplit(sep = '.')[0])

        v = pd.read_csv(file).values.T[0]

        n = v.shape[0] - 1
        w = v[0:n]
        b = v[n]

        n_data = X.shape[0]
        for data in range(n_data):
            f = y[data] * (np.dot(w, X[data]) + b)
            if f > 0:
                guess_true += 1
        acc_dict[iter] = guess_true/n_data

    acc   = []
    iters = np.array(list(acc_dict.keys()))
    iters = np.sort(iters, axis = None)
    print(iters)
    for i in range(iters.shape[0]):
        acc.append(acc_dict[iters[i]])
    acc = np.array(acc)

    return [iters, acc]

