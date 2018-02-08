import numpy as np
import pandas as pd
import subprocess as sub
from kernels import rbf
from .Network import Network
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedKFold
from pathconf import params_path, datas_path, non_linear_mpi_path, results_path, bad_chess_path

class NDSVM(object):
    def __init__(self, nodes, C = 60, c = 10, gamma = 2**-15, p = 100, max_iter = 400, step = 10):
        self.C        = C
        self.c        = c
        self.p        = p
        self.gamma    = gamma
        self.nodes    = nodes
        self.max_iter = max_iter
        self.step      = step
        self.network  = Network(self.nodes)

        self.network.create_graph_mpi()
        params = pd.DataFrame(data  = [[self.C], [self.c], [self.gamma], [self.max_iter], [self.step]],
                              index = ['C', 'c', 'gamma', 'max_iter', 'step'])
        params.to_csv(params_path)

    def get_nodes(self):
        return self.nodes

    def get_iters(self):
        iters    = []
        for i in range(int(self.max_iter/self.step)):
            iters.append(self.step * (i + 1))
        return np.array(iters)

    def get_local_classifier_iter(self, node, i):
        file_alpha = results_path.joinpath("alpha_" + str(i) + "_" + str(node) + ".csv")
        file_beta  = results_path.joinpath("beta_" + str(i) + "_" + str(node) + ".csv")
        file_b     = results_path.joinpath("b_" + str(i) + "_" + str(node) + ".csv")
        alpha      = pd.read_csv(file_alpha).values.T[0]
        beta       = pd.read_csv(file_beta).values.T[0]
        b          = pd.read_csv(file_b).values[0][0]

        return [alpha, beta, b]

    def get_best_local_classifier(self, node):
        iters = self.get_iters()
        return self.get_local_classifier_iter(node, iters[-1])

    def set_params(self, C, c, gamma, p, max_iter = 400, step = 5):
        self.clean_files()

        self.C        = C
        self.c        = c
        self.p        = p
        self.gamma    = gamma
        self.max_iter = max_iter
        self.step     = step
        params        = pd.DataFrame(data  = [[self.C], [self.c], [self.gamma], [self.max_iter], [self.step]],
                                     index = ['C', 'c', 'gamma', 'max_iter', 'step'])
        params.to_csv(params_path)

    def create_commun_data(self, X):
        n_data, dim = X.shape
        max_att     = X[0]
        min_att     = X[0]
        for i in range(1, n_data):
            max_att = np.maximum(max_att, X[i])
            min_att = np.minimum(min_att, X[i])
        delta = max_att - min_att

        random   = np.random.RandomState()
        rand_vec = random.uniform(size = self.p * dim)

        chi = []
        for i in range(self.p):
            start = i * dim
            line = rand_vec[start:start + dim] * delta + min_att
            chi.append(line)

        chi = pd.DataFrame(np.array(chi))
        chi.to_csv(datas_path.joinpath("chi.csv"), index = False)

    def fit(self, X, y, stratified = True, bad_chess = False):
        self.network.split_data(X, y, stratified, bad_chess)
        self.create_commun_data(X)
        command = "mpiexec -n " + str(self.nodes) + " python " + str(non_linear_mpi_path)
        sub.check_call(command, shell = True)

    def local_discriminant(self, node, alpha, beta, b, x):
        X_file   = str(datas_path) + "/data_" + str(node) + ".csv"
        X        = pd.read_csv(X_file).values
        chi_file = str(datas_path) + "/chi.csv"
        chi      = pd.read_csv(chi_file).values

        g = b
        n_node_data = X.shape[0]
        n_chi_data  = chi.shape[0]
        for i in range(n_node_data):
            g += alpha[i] * rbf(x, X[i], self.gamma)
        for i in range(n_chi_data):
            g += beta[i] * rbf(x, chi[i], self.gamma)

        return np.sign(g)

    def mix_discriminant(self, x):
        for node in range(self.nodes):
            alpha, beta, b = self.get_best_local_classifier(node)
            guess  = 0
            if self.local_discriminant(node, alpha, beta, b, x) > 0:
                guess += 1
            else:
                guess += -1

        return np.sign(guess)

    def local_score_iter(self, X, y, node, i):
        alpha, beta, b = self.get_local_classifier_iter(node, i)
        guess_true     = 0
        n_data         = X.shape[0]
        for data in range(n_data):
            if y[data] * self.local_discriminant(node, alpha, beta, b, X[data]) > 0:
                guess_true += 1

        return guess_true/n_data

    def local_best_score(self, X, y, node):
        iters = self.get_iters()
        return self.local_score_iter(X, y, node, iters[-1])

    def all_local_iters_risk(self, node, X, y):
        acc   = []
        iters = self.get_iters()
        for i in iters:
            acc.append(self.local_score_iter(X, y, node, i))
        acc = np.array(acc)

        return 1 - acc

    def mix_score(self, X, y):
        guess_true = 0
        n_data     = X.shape[0]
        for data in range(n_data):
            if y[data] * self.mix_discriminant(X[data]) > 0:
                guess_true += 1

        return guess_true/n_data

    def local_grid_search(self, X, y, params, node, stratified = True, scale = True):
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
                acc += self.local_best_score(X_test, y_test, node)
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
