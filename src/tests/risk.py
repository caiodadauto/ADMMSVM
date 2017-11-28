import numpy as np
import pandas as pd
import analysisdata as analysis
from pathconf import datas_path
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

def risk(dsvm, X, y):
    analysis.visualization(X, y)

    tests = {
            "$C = 2^{-15}\;\;\mathrm{e}\;\;c = 1$"   : {'C': 2**-15, 'c': 1},
            "$C = 2^{-15}\;\;\mathrm{e}\;\;c = 10$"  : {'C': 2**-15, 'c': 10},
            "$C = 2^{-5}\;\;\mathrm{e}\;\;c = 1$"    : {'C': 2**-5, 'c': 1},
            "$C = 2^{-5}\;\;\mathrm{e}\;\;c = 10$"   : {'C': 2**-5, 'c': 10}
            }

    params_local_central = {
            "C"        : [2**-15, 2**-10, 2**-5, 2],
            "max_iter" : [400],
            "penalty"  : ['l1'],
            "dual"     : [False]
            }


    risk_local   = 0
    risk_central = 0
    risk_dist    = {
            "$C = 2^{-15}\;\;\mathrm{e}\;\;c = 1$"   : [],
            "$C = 2^{-15}\;\;\mathrm{e}\;\;c = 10$"  : [],
            "$C = 2^{-5}\;\;\mathrm{e}\;\;c = 1$"    : [],
            "$C = 2^{-5}\;\;\mathrm{e}\;\;c = 10$"   : []
            }
    skf = StratifiedKFold()
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for test, params in tests.items():
            dsvm.set_params(**params)
            dsvm.fit(X_train, y_train)
            risk = dsvm.all_iters_risk(X_test, y_test)
            risk_dist[test].append(risk)

        local_data    = str(datas_path) + "/data_0.csv"
        local_class   = str(datas_path) + "/class_0.csv"
        X_local_train = pd.read_csv(local_data).values
        y_local_train = pd.read_csv(local_class).values.T[0]
        scale         = StandardScaler().fit(X_local_train)
        X_local_train = scale.transform(X_local_train)
        X_local_test  = scale.transform(X_test)
        scale         = StandardScaler().fit(X_train)
        X_train       = scale.transform(X_train)
        X_test        = scale.transform(X_test)

        gs             = GridSearchCV(LinearSVC(), params_local_central)
        gs.fit(X_local_train, y_local_train)
        local_params   = gs.best_params_
        gs.fit(X_train, y_train)
        central_params = gs.best_params_

        local_model  = LinearSVC(**local_params).fit(X_local_train, y_local_train)
        risk_local  += 1 - local_model.score(X_local_test, y_test)

        central_model = LinearSVC(**central_params).fit(X_train, y_train)
        risk_central  += 1 - central_model.score(X_test, y_test)

    for test in risk_dist.keys():
        risk_dist[test] = np.array(risk_dist[test]).sum(axis = 0)/3
    risk_local   /= 3
    risk_central /= 3
    analysis.plot_risk(risk_local, risk_central, risk_dist, dsvm.get_iters())
