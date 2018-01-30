import numpy as np
import pandas as pd
import analysisdata as analysis
from pathconf import datas_path
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

def cancer(ndsvm, X, y):
    tests = {
            "$C = 1,\;\;c = 1,\;\;\gamma = 0.7\;\;\mathrm{e}\;\;p = 200$"   : {
                'C': 60, 'c':1, 'gamma': 0.7, 'p': 200, 'max_iter': 800
                },
            "$C = 1,\;\;c = 1,\;\;\gamma = 0.7\;\;\mathrm{e}\;\;p = 300$"   : {
                'C': 60, 'c':1, 'gamma': 0.7, 'p': 300, 'max_iter': 800
                },
            "$C = 1,\;\;c = 1,\;\;\gamma = 0.7\;\;\mathrm{e}\;\;p = 500$"  : {
                'C': 60, 'c': 1, 'gamma': 0.7, 'p': 500, 'max_iter': 800
                },
            "$C = 1,\;\;c = 1,\;\;\gamma = 0.7\;\;\mathrm{e}\;\;p = 800$"  : {
                'C': 60, 'c': 1, 'gamma': 0.7, 'p': 800, 'max_iter': 800
                }
            }

    params_local_central = {
            "C"        : [1, 60],
            "gamma"    : [2**-15, 0.7],
            "max_iter" : [800],
            }


    risk_local   = 0
    risk_central = 0
    risk_dist    = {
            "$C = 1,\;\;c = 1,\;\;\gamma = 0.7\;\;\mathrm{e}\;\;p = 200$"  : [],
            "$C = 1,\;\;c = 1,\;\;\gamma = 0.7\;\;\mathrm{e}\;\;p = 300$"  : [],
            "$C = 1,\;\;c = 1,\;\;\gamma = 0.7\;\;\mathrm{e}\;\;p = 500$"  : [],
            "$C = 1,\;\;c = 1,\;\;\gamma = 0.7\;\;\mathrm{e}\;\;p = 800$"  : []
            }

    gs     = GridSearchCV(SVC(), params_local_central)
    scaler = StandardScaler()
    skf    = StratifiedKFold()
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        ndsvm.network.split_data(X_train, y_train)
        local_data          = str(datas_path) + "/data_0.csv"
        local_class         = str(datas_path) + "/class_0.csv"
        X_local_train       = pd.read_csv(local_data).values
        y_local_train       = pd.read_csv(local_class).values.T[0]
        X_local_train_scale = scaler.fit_transform(X_local_train)
        X_local_test_scale  = scaler.transform(X_test)
        X_train_scale       = scaler.fit_transform(X_train)
        X_test_scale        = scaler.transform(X_test)

        for test, params in tests.items():
            ndsvm.set_params(**params)
            ndsvm.fit(X_train_scale, y_train)
            risks = ndsvm.all_iters_risk(0, X_test_scale, y_test)
            risk_dist[test].append(risks)

        gs.fit(X_local_train, y_local_train)
        local_params   = gs.best_params_
        gs.fit(X_train, y_train)
        central_params = gs.best_params_

        local_model   = SVC(**local_params).fit(X_local_train_scale, y_local_train)
        central_model = SVC(**central_params).fit(X_train_scale, y_train)

        risk_local    += (1 - local_model.score(X_local_test_scale, y_test))/3
        risk_central  += (1 - central_model.score(X_test_scale, y_test))/3

        print("Final Iteration")

    for test in risk_dist.keys():
        risk_dist[test] = np.array(risk_dist[test]).sum(axis = 0)/3
    analysis.plot_risk(risk_local, risk_central, risk_dist, ndsvm.get_iters())
