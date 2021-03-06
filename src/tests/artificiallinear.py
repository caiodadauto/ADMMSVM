import numpy as np
import pandas as pd
import analysisdata as analysis
from pathconf import datas_path
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

def get_risks(ldsvm, params_svm, params_dist_svm, X, y):
    local_risk   = 0
    central_risk = 0
    ldsvm_risk   = 0

    gs     = GridSearchCV(LinearSVC(), params_svm)
    scaler = StandardScaler()
    skf    = StratifiedKFold()
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        ldsvm.network.split_data(X_train, y_train, stratified=False)
        local_data          = str(datas_path) + "/data_0.csv"
        local_class         = str(datas_path) + "/class_0.csv"
        X_local_train       = pd.read_csv(local_data).values
        y_local_train       = pd.read_csv(local_class).values.T[0]
        X_local_train_scale = scaler.fit_transform(X_local_train)
        X_local_test_scale  = scaler.transform(X_test)
        X_train_scale       = scaler.fit_transform(X_train)
        X_test_scale        = scaler.transform(X_test)

        params_dist_best = ldsvm.grid_search(X_train, y_train, params_dist_svm, stratified=False)
        gs.fit(X_local_train, y_local_train)
        params_local_best = gs.best_params_
        gs.fit(X_train, y_train)
        params_central_best = gs.best_params_

        ldsvm.set_params(**params_dist_best)
        ldsvm.fit(X_train_scale, y_train, stratified=False)
        local_model   = LinearSVC(**params_local_best).fit(X_local_train_scale, y_local_train)
        central_model = LinearSVC(**params_central_best).fit(X_train_scale, y_train)

        ldsvm_risk   += (1 - ldsvm.score(X_test_scale, y_test))/3
        local_risk   += (1 - local_model.score(X_local_test_scale, y_test))/3
        central_risk += (1 - central_model.score(X_test_scale, y_test))/3
    return [local_risk, central_risk, ldsvm_risk]

def artificial_linear(ldsvm, X, y):
    params_dist_svm = {
            "C"        : [1, 2**2, 5, 2**3, 2**4],
            "c"        : [5, 10, 12, 16, 18],
            "max_iter" : [400],
            "step"     : [10]
            }


    params_svm = {
            "C"        : [0.1, 0.3, 0.5, 1, 2, 3, 6, 2**3],
            "max_iter" : [400],
            "penalty"  : ['l1'],
            "dual"     : [False]
            }

    local_risk, central_risk, ldsvm_risk = get_risks(ldsvm, params_svm, params_dist_svm, X, y)

    print(">-------------Best Risks from Grid Search---------------------<")
    print("Risk Local         --> ", local_risk)
    print("Risk LDSVM         --> ", ldsvm_risk)
    print("Risk Central       --> ", central_risk)

    gs      = GridSearchCV(LinearSVC(), params_svm)
    scaler  = StandardScaler()

    ldsvm.network.split_data(X, y, stratified=False)
    local_data    = str(datas_path) + "/data_0.csv"
    local_class   = str(datas_path) + "/class_0.csv"
    X_local       = pd.read_csv(local_data).values
    y_local       = pd.read_csv(local_class).values.T[0]
    X_local_scale = scaler.fit_transform(X_local)
    X_scale       = scaler.fit_transform(X)

    params_dist_best = ldsvm.grid_search(X, y, params_dist_svm, stratified=False)
    gs.fit(X_local, y_local)
    params_local_best = gs.best_params_
    gs.fit(X, y)
    params_central_best = gs.best_params_

    ldsvm.set_params(**params_dist_best)
    ldsvm.fit(X_scale, y, stratified=False)
    local_model   = LinearSVC(**params_local_best).fit(X_local_scale, y_local)
    central_model = LinearSVC(**params_central_best).fit(X_scale, y)

    print(">-------------Best Parameters for Whole data Set--------------<")
    print("Parameters Local   --> ", params_local_best)
    print("Parameters LDSVM   --> ", params_dist_best)
    print("Parameters Central -->", params_central_best)

    analysis.plot_planes(X, y, local_model, central_model, ldsvm)
    analysis.plot_dispersion(ldsvm)
