import numpy as np
import pandas as pd
import analysisdata as analysis
from pathconf import datas_path
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

def get_risks(dsvm, params_svm, params_dist_svm, X, y):
    local_risk            = 0
    central_risk          = 0
    dsvm_risk             = 0

    gs     = GridSearchCV(LinearSVC(), params_svm)
    scaler = StandardScaler()
    skf    = StratifiedKFold()
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train_scale = scaler.fit_transform(X_train)
        X_test_scale  = scaler.transform(X_test)
        dsvm.network.split_data(X_train, y_train, stratified=False)
        local_data  = str(datas_path) + "/data_0.csv"
        local_class = str(datas_path) + "/class_0.csv"
        X_local_train = pd.read_csv(local_data).values
        y_local_train = pd.read_csv(local_class).values.T[0]
        X_local_train_scale = scaler.fit_transform(X_local_train)

        params_dist_best = dsvm.grid_search(X_train, y_train, params_dist_svm, stratified=False)
        gs.fit(X_local_train, y_local_train)
        params_local_best = gs.best_params_
        gs.fit(X_train, y_train)
        params_central_best = gs.best_params_

        dsvm.set_params(**params_dist_best)
        dsvm.fit(X_train_scale, y_train, stratified=False)
        local_model   = LinearSVC(**params_local_best).fit(X_local_train_scale, y_local_train)
        central_model = LinearSVC(**params_central_best).fit(X_train_scale, y_train)

        dsvm_risk    += (1 - dsvm.score(X_test_scale, y_test))/3
        local_risk   += (1 - local_model.score(X_test_scale, y_test))/3
        central_risk += (1 - central_model.score(X_test_scale, y_test))/3
    return [local_risk, central_risk, dsvm_risk]

def plane(dsvm, X, y):
    params_dist_svm = {
            "C"        : [2, 2**4, 2**5, 2**6],#[32],
            "c"        : [10, 15, 18, 20, 22, 25],#[20],
            "max_iter" : [250],
            "step"     : [50]
            }


    params_svm = {
            "C"        : [2**-7, 2**-6, 2**-5, 0.5, 1, 2, 3, 2**5, 2**6, 2**7],#[0.03125, 2, 1024],
            "max_iter" : [250],
            "penalty"  : ['l1'],
            "dual"     : [False]
            }

    local_risk, central_risk, dsvm_risk = get_risks(dsvm, params_svm, params_dist_svm, X, y)
    print("local --> ", local_risk)
    print("central --> ", central_risk)
    print("dsvm --> ", dsvm_risk)

    gs      = GridSearchCV(LinearSVC(), params_svm)
    scaler  = StandardScaler()

    X_scale = scaler.fit_transform(X)
    dsvm.network.split_data(X, y, stratified=False)
    local_data    = str(datas_path) + "/data_0.csv"
    local_class   = str(datas_path) + "/class_0.csv"
    X_local       = pd.read_csv(local_data).values
    y_local       = pd.read_csv(local_class).values.T[0]
    X_local_scale = scaler.fit_transform(X_local)

    params_dist_best = dsvm.grid_search(X, y, params_dist_svm, stratified=False)
    print("DSVM params --> ", params_dist_best)
    gs.fit(X_local, y_local)
    params_local_best = gs.best_params_
    print("local params --> ", params_local_best)
    gs.fit(X, y)
    params_central_best = gs.best_params_
    print("central params-->", params_central_best)

    dsvm.set_params(**params_dist_best)
    dsvm.fit(X_scale, y, stratified=False)
    local_model   = LinearSVC(**params_local_best).fit(X_local_scale, y_local)
    central_model = LinearSVC(**params_central_best).fit(X_scale, y)

    analysis.plot_planes(X, y, local_model, central_model, dsvm)
