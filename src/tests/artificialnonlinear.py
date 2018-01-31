import numpy as np
import pandas as pd
import analysisdata as analysis
from pathconf import datas_path, results_path, graph_path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

def get_risks(ndsvm, params_svm, params_dist_svm, X, y):
    local_risk   = 0
    central_risk = 0
    ldsvm_risk   = np.zeros(ndsvm.get_nodes())

    gs     = GridSearchCV(LinearSVC(), params_svm)
    scaler = StandardScaler()
    skf    = StratifiedKFold()
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        ndsvm.network.split_data(X_train, y_train, stratified=False)
        local_data          = str(datas_path) + "/data_0.csv"
        local_class         = str(datas_path) + "/class_0.csv"
        X_local_train       = pd.read_csv(local_data).values
        y_local_train       = pd.read_csv(local_class).values.T[0]
        X_local_train_scale = scaler.fit_transform(X_local_train)
        X_local_test_scale  = scaler.transform(X_test)
        X_train_scale       = scaler.fit_transform(X_train)
        X_test_scale        = scaler.transform(X_test)

        for n in range(ndsvm.get_nodes()):
            params_dist_best = ndsvm.local_grid_search(X_train, y_train, params_dist_svm, n)
            ndsvm.set_params(**params_dist_best)
            ndsvm.fit(X_train_scale, y_train)
            ldsvm_risk[n] += (1 - ldsvm.local_best_score(X_test_scale, y_test, n))/3

        gs.fit(X_local_train, y_local_train)
        params_local_best = gs.best_params_
        gs.fit(X_train, y_train)
        params_central_best = gs.best_params_

        local_model   = LinearSVC(**params_local_best).fit(X_local_train_scale, y_local_train)
        central_model = LinearSVC(**params_central_best).fit(X_train_scale, y_train)

        local_risk   += (1 - local_model.score(X_local_test_scale, y_test))/3
        central_risk += (1 - central_model.score(X_test_scale, y_test))/3

    return [local_risk, central_risk, ldsvm_risk]

def artificial_non_linear(ndsvm, X, y):
    params_dist_svm = {
            "C"        : [1, 32, 64],#
            "c"        : [1, 8],#
            "gamma"    : [2**-5, 2**-3, 2**-2, 2**-1, 1],
            "p"        : [150],
            "max_iter" : [800],
            "step"     : [800]
            }


    params_svm = {
            "C"        : [1, 2, 4, 8, 16, 32, 64],
            "gamma"    : [2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 1],
            }

    local_risk, central_risk, ndsvm_risk = get_risks(ndsvm, params_svm, params_dist_svm, X, y)

    print(">-------------Best Risks from Grid Search---------------------<")
    print("Risk Local         --> ", local_risk)
    print("Risk NDSVM         --> ", ndsvm_risk)
    print("Risk Central       --> ", central_risk)

    scaler  = StandardScaler()
    ndsvm.network.split_data(X, y, stratified=False)
    local_data    = str(datas_path) + "/data_0.csv"
    local_class   = str(datas_path) + "/class_0.csv"
    X_local       = pd.read_csv(local_data).values
    y_local       = pd.read_csv(local_class).values.T[0]
    X_local_scale = scaler.fit_transform(X_local)
    X_scale       = scaler.fit_transform(X)

    print(">-------------Best Parameters for Whole data Set--------------<")
    for n in range(ndsvm.get_nodes()):
        params_dist_best = ndsvm.local_grid_search(X, y, params_dist_svm, n)
        print("Parameters NDSVM  node " + str(n + 1) + " --> ", params_dist_best)
        ndsvm.set_params(**params_dist_best)
        ndsvm.fit(X_scale, y)
        analysis.plot_non_linear_classifier(ndsvm, X, y, "dist_non_linear_classifier_" + str(n), node = n)

    gs.fit(X_local, y_local)
    params_local_best = gs.best_params_
    gs.fit(X, y)
    params_central_best = gs.best_params_
    print("Parameters Local         --> ", params_local_best)
    print("Parameters Central       -->", params_central_best)

    local_model   = SVC(**params_local_best).fit(X_local_scale, y_local)
    central_model = SVC(**params_central_best).fit(X_scale, y)
    analysis.plot_non_linear_classifier(local_model, X, y, "local_non_linear_classifier", dist = False)
    analysis.plot_non_linear_classifier(central_model, X, y, "central_non_linear_classifier", dist = False)
