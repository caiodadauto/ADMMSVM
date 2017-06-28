import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import SparsePCA
from pathlib import Path

def plot_accuracy(acc_local, acc_central):
    pathlist = Path('partialresults/').glob('*.csv')
    for path in pathlist:
        path_str = str(path)
        iter     = int(path_str.rsplit(sep = '_')[-1].rsplit(sep = '.')[0])

        v = pd.read_csv(path_str).values.T[0]
        n = v.shape[0] - 1
        w = v[0:n]
        b = v[n]

        w = pca.transform(np.array([w]))
        print(w)
        print(b)

        plt.style.use('seaborn')
        plt.figure(figsize=(8, 8))

        colors = []
        for l in y:
            if l > 0:
                colors.append('b')
            else:
                colors.append('r')

        plt.scatter(X.T[0], X.T[1], c = colors, alpha = 0.65)

        '''
        plt.xlim(-5000,5000)
        plt.ylim(-2000,2000)
        '''

        plt.show()



