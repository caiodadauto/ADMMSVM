import numpy as np
import pandas as pd
import subprocess as sub
from pathlib import Path
from .Network import Network

class DistSVM(object):
    def __init__(self, nodes, C = 60, c = 10, max_iter = 200):
        self.nodes      = nodes
        self.C          = C
        self.c          = c
        self.max_iter   = max_iter
        self.network    = Network(nodes = self.nodes)

        self.network.create_graph_mpi('src/distributedsvm/graph/graph_mpi.csv')
        params = pd.DataFrame(data  = [[self.C], [self.c], [self.max_iter]],
                              index = ['C', 'c', 'max_iter'])
        params.to_csv('src/distributedsvm/params.csv')

    def get_nodes(self):
        return self.node

    def set_params(self, C, c):
        self.C = C
        self.c = c
        params = pd.DataFrame(data  = [[self.C], [self.c], [self.max_iter]],
                              index = ['C', 'c', 'max_iter'])
        params.to_csv('src/distributedsvm/params.csv')

    def fit(self, X, y):
        self.network.split_data(X, y, 'src/distributedsvm/datas')
        command = "mpiexec -n " + str(self.nodes) + " python src/distributedsvm/mpi.py"
        sub.check_call(command, shell = True)

    def risk_score(self, X, y):
        acc_dict = {}
        pathlist = Path('src/distributedsvm/results/').glob('*.csv')
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
        for i in range(iters.shape[0]):
            acc.append(acc_dict[iters[i]])
        acc = np.array(acc)

        return [iters, 1 - acc]

    def __del__(self):
        sub.check_call('rm src/distributedsvm/params.csv', shell = True)
        sub.check_call('rm src/distributedsvm/results/*.csv', shell = True)
