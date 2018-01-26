import numpy as np
import pandas as pd
import analysisdata as analysis
from pathconf import datas_path, results_path, graph_path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

def get_risks(ndsvm, params_svm, params_dist_svm, X, y):
    central_risk          = 0
    ndsvm_risk            = 0

    gs     = GridSearchCV(SVC(), params_svm)
    scaler = StandardScaler()
    skf    = StratifiedKFold()
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train_scale       = scaler.fit_transform(X_train)
        X_test_scale        = scaler.transform(X_test)

        params_dist_best = ndsvm.grid_search(X_train, y_train, params_dist_svm)
        gs.fit(X_train, y_train)
        params_central_best = gs.best_params_

        ndsvm.set_params(**params_dist_best)
        ndsvm.fit(X_train_scale, y_train)
        central_model = SVC(**params_central_best).fit(X_train_scale, y_train)

        ndsvm_risk   += (1 - ndsvm.mix_score(X_test_scale, y_test))/3
        central_risk += (1 - central_model.score(X_test_scale, y_test))/3
    return [central_risk, ndsvm_risk]

def chess(ndsvm, X, y):
    params_dist_svm = {
            "C"        : 1,#
            "c"        : 2,#
            "gamma"    : 0.7,
            "p"        : 250,
            "max_iter" : 400
            }


    params_svm = {
            "C"        : [2**-5,2**-2,2,2**2],#
            "gamma"    : [2**-15,2**-10,2**-5,2],
            "max_iter" : [200]
            }

    # local_risk, central_risk, ndsvm_risk = get_risks(ndsvm, params_svm, params_dist_svm, X, y)
    #
    # print(">-------------Estimativas para o Risco---------------------<")
    # print("Risco Local   --> ", local_risk)
    # print("Risco DSVM    --> ", ndsvm_risk)
    # print("Risco Central --> ", central_risk)

    # gs      = GridSearchCV(SVC(), params_svm)
    scaler  = StandardScaler()
    X_scale = scaler.fit_transform(X)

    # params_dist_best = ndsvm.grid_search(X, y, params_dist_svm)
    # gs.fit(X, y)
    # params_central_best = gs.best_params_

    # print(">--------------------Melhores Parametros para o Conjunto Total-------------------<")
    # print("Parametros Local   --> ", params_local_best)
    # print("Parametros DSVM    --> ", params_dist_best)
    # print("Parametros Central -->", params_central_best)

    ndsvm.set_params(C=1,c=1,gamma=0.7,p=250, max_iter=400)
    ndsvm.fit(X_scale, y, bad_chess = False)
    # central_model = SVC(**params_central_best).fit(X_scale, y)

    # command = "cp " + str(results_path) + "/*.csv " + str(graph_path)

    analysis.plot_non_linear_classifier(ndsvm, 0, X, y)
