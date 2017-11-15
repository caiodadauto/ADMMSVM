import numpy as np
import pandas as pd
import analysisdata as analysis
from pathconf import datas_path
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

def plane(dsvm, X, y):
    params_dist_svm = {
            "C"        : [32], #[2**-15, 2**-10, 2**-5, 2, 2**5, 2**10],
            "c"        : [20], #[0.5, 1, 10, 20, 100],
            "max_iter" : [250],
            "step"     : [50]
            }


    params_svm = {
            "C"        : [0.03125, 2, 1024], #[2**-15, 2**-10, 2**-5, 2, 2**5, 2**10],
            "max_iter" : [250],
            "penalty"  : ['l1'],
            "dual"     : [False]
            }

    params_dist_best = dsvm.grid_search(X, y, params_dist_svm)
    dsvm.set_params(**params_dist_best)
    dsvm.fit(X, y)

    local_data  = str(datas_path) + "/data_0.csv"
    local_class = str(datas_path) + "/class_0.csv"

    X_local_stratified = pd.read_csv(local_data).values
    y_local_stratified = pd.read_csv(local_class).values.T[0]

    dsvm.network.split_data(X, y, False)
    X_local = pd.read_csv(local_data).values
    y_local = pd.read_csv(local_class).values.T[0]

    gs = GridSearchCV(LinearSVC(), params_svm)

    gs.fit(X_local, y_local)
    params_local_best = gs.best_params_

    gs.fit(X_local_stratified, y_local_stratified)
    params_local_best_stratified = gs.best_params_

    gs.fit(X, y)
    params_central_best = gs.best_params_

    local_model              = LinearSVC(**params_local_best).fit(X_local, y_local)
    local_model_stratified   = LinearSVC(**params_local_best_stratified).fit(X_local_stratified, y_local_stratified)
    central_model            = LinearSVC(**params_central_best).fit(X, y)

    analysis.plot_planes(X, y, local_model, local_model_stratified, central_model, dsvm)
