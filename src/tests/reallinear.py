import numpy as np
import pandas as pd
import analysisdata as analysis
from pathconf import datas_path
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

def real_linear(ldsvm, X, y):
    tests = {
            "$C = 2^{-5}\;\;\mathrm{e}\;\;c = 1$"   : {'C': 2**-5, 'c':1 },
            "$C = 2^{-5}\;\;\mathrm{e}\;\;c = 10$"   : {'C': 2**-5, 'c':10 },
            "$C = 2^{-2}\;\;\mathrm{e}\;\;c = 1$"  : {'C': 2**-2, 'c': 1},
            "$C = 2^{-2}\;\;\mathrm{e}\;\;c = 10$"  : {'C': 2**-2, 'c': 10}
            }

    params_local_central = {
            "C"        : [2**-5, 2**-2, 2, 2**2, 2**5],
            "max_iter" : [400],
            "penalty"  : ['l1'],
            "dual"     : [False]
            }


    risk_local   = 0
    risk_central = 0
    risk_dist    = {
            "$C = 2^{-5}\;\;\mathrm{e}\;\;c = 1$"   : [],
            "$C = 2^{-5}\;\;\mathrm{e}\;\;c = 10$"      : [],
            "$C = 2^{-2}\;\;\mathrm{e}\;\;c = 1$"   : [],
            "$C = 2^{-2}\;\;\mathrm{e}\;\;c = 10$"   : []
            }

    gs     = GridSearchCV(LinearSVC(), params_local_central)
    scaler = StandardScaler()
    skf    = StratifiedKFold()
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        ldsvm.network.split_data(X_train, y_train)
        local_data          = str(datas_path) + "/data_0.csv"
        local_class         = str(datas_path) + "/class_0.csv"
        X_local_train       = pd.read_csv(local_data).values
        y_local_train       = pd.read_csv(local_class).values.T[0]
        X_local_train_scale = scaler.fit_transform(X_local_train)
        X_local_test_scale  = scaler.transform(X_test)
        X_train_scale       = scaler.fit_transform(X_train)
        X_test_scale        = scaler.transform(X_test)

        for test, params in tests.items():
            ldsvm.set_params(**params)
            ldsvm.fit(X_train_scale, y_train)
            risks = ldsvm.all_iters_risk(X_test_scale, y_test)
            risk_dist[test].append(risks)

        gs.fit(X_local_train, y_local_train)
        local_params   = gs.best_params_
        gs.fit(X_train, y_train)
        central_params = gs.best_params_

        local_model   = LinearSVC(**local_params).fit(X_local_train_scale, y_local_train)
        central_model = LinearSVC(**central_params).fit(X_train_scale, y_train)

        risk_local    += (1 - local_model.score(X_local_test_scale, y_test))/3
        risk_central  += (1 - central_model.score(X_test_scale, y_test))/3

    for test in risk_dist.keys():
        risk_dist[test] = np.array(risk_dist[test]).sum(axis = 0)/3
    analysis.plot_risk(risk_local, risk_central, risk_dist, ldsvm.get_iters(),
            name_file = 'risk_plot', label_x = 'Iterações ' + r'$(l)$', label_y = 'Risco Médio')
