import numpy as np
import pandas as pd
import subprocess as sub
from .Network import Network
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedKFold
from pathconf import params_path, datas_path, mpi_path, results_path

class DSVM(object):
    def __init__(self, nodes, C = 60, c = 10, max_iter = 400, step = 5):
        self.C        = C
        self.c        = c
        self.step     = step
        self.nodes    = nodes
        self.max_iter = max_iter
        self.network  = Network(self.nodes)

        self.network.create_graph_mpi()
        params = pd.DataFrame(data  = [[self.C], [self.c], [self.max_iter], [self.step]],
                              index = ['C', 'c', 'max_iter', 'step'])
        params.to_csv(params_path)

    def get_nodes(self):
        return self.nodes

    def get_iters(self):
        iters    = []
        pathlist = sorted(results_path.glob('*_0.csv'))
        for path in pathlist:
            iters.append(int(path.stem.rsplit(sep = '_')[2]))
        iters.sort()
        return np.array(iters)


    def get_plane(self, file):
        v          = pd.read_csv(file).values.T[0]
        n          = v.shape[0] - 1
        w          = v[0:n]
        b          = v[n]

        return [w, b]

    def get_best_plane(self, node):
        file =  results_path.joinpath('(w,b)_partial_' + str(self.get_iters()[-1]) + "_" + str(node) + ".csv")
        return self.get_plane(file)

    def get_all_planes(self):
        planes_per_node = []
        for node in range(self.nodes):
            node_planes = []
            pathlist = sorted(results_path.glob('*_' + str(node) + '.csv'), key = lambda a: int(a.stem.rsplit(sep = '_')[2]))
            for path in pathlist:
                node_planes.append(self.get_plane(path))
            planes_per_node.append(node_planes)

        return planes_per_node

    def set_params(self, C, c, max_iter = 400, step = 5):
        self.clean_files()

        self.C        = C
        self.c        = c
        self.step     = step
        self.max_iter = max_iter
        params        = pd.DataFrame(data  = [[self.C], [self.c], [self.max_iter], [self.step]],
                                     index = ['C', 'c', 'max_iter', 'step'])
        params.to_csv(params_path)

    def fit(self, X, y, stratified = True):
        self.network.split_data(X, y, stratified)
        command = "mpiexec -n " + str(self.nodes) + " python " + str(mpi_path)
        sub.check_call(command, shell = True)

    def score(self, X, y):
        score = 0
        for node in range(self.nodes):
            w, b = self.get_best_plane(node)
            score += self.score_from_plane(X, y, w, b)
        return score/self.nodes

    def score_from_plane(self, X, y, w, b):
        guess_true = 0
        n_data     = X.shape[0]
        for data in range(n_data):
            f = y[data] * (np.dot(w, X[data]) + b)
            if f > 0:
                guess_true += 1

        return guess_true/n_data

    def all_iters_risk(self, X, y):
        acc             = []
        planes_per_node = self.get_all_planes()
        for planes in planes_per_node:
            node_acc = []
            for plane in planes:
                node_acc.append(self.score_from_plane(X, y, plane[0], plane[1]))
            acc.append(node_acc)
        acc = np.array(acc).sum(axis = 0)/self.nodes

        return 1 - acc

    def grid_search(self, X, y, params, stratified = True, scale = True):
        max_acc  = 0
        skf      = StratifiedKFold(n_splits = 2)
        grid     = ParameterGrid(params)
        for param in grid:
            acc = 0
            self.set_params(**param)
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                if scale:
                    scaler  = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test  = scaler.transform(X_test)
                self.fit(X_train, y_train, stratified)
                acc += self.score(X_test, y_test)
            acc /= 2
            if max_acc < acc:
                best_params = param
                max_acc     = acc

        return best_params

    def clean_files(self):
        if params_path.exists:
            command = "rm " + str(params_path)
            sub.check_call(command, shell = True)
        if list(results_path.glob('*.csv')):
            command = "rm " + str(results_path) + "/*.csv"
            sub.check_call(command, shell = True)

    def __del__(self):
        self.clean_files()
