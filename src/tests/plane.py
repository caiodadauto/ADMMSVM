import numpy as np
import pandas as pd
import src.distributedsvm as dist
import src.analysisdata as analysis
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

def plane(distSVM, X, y):
    params_dist_svm = {"C"    : [32], #[2**-15, 2**-10, 2**-5, 2, 2**5, 2**10],
                       "c"    : [20], #[0.5, 1, 10, 20, 100],
                       "max_iter": [250],
                       "step" : [50]
                      }


    params_svm = {"C" : [0.03125, 2, 1024], #[2**-15, 2**-10, 2**-5, 2, 2**5, 2**10],
                  "max_iter" : [250],
                  "penalty"  : ['l1'],
                  "dual"     : [False]
                 }

    params_dist_best = distSVM.grid_search(X, y, params_dist_svm)
    print("dist  ", params_dist_best)
    distSVM.set_params(**params_dist_best)
    distSVM.fit(X, y)

    X_local = pd.read_csv('src/distributedsvm/datas/data_0.csv').values
    y_local = pd.read_csv('src/distributedsvm/datas/class_0.csv').values.T[0]

    distSVM.network.split_data(X, y, True)
    X_local_stratified = pd.read_csv('src/distributedsvm/datas/data_0.csv').values
    y_local_stratified = pd.read_csv('src/distributedsvm/datas/class_0.csv').values.T[0]

    gs = GridSearchCV(LinearSVC(), params_svm)

    gs.fit(X_local, y_local)
    params_local_best = gs.best_params_
    print("local  ", params_local_best)

    gs.fit(X_local_stratified, y_local_stratified)
    params_local_best_stratified = gs.best_params_
    print("local estrat  ", params_local_best_stratified)

    gs.fit(X, y)
    params_central_best = gs.best_params_
    print("central  ", params_central_best)

    local_model              = LinearSVC(**params_local_best).fit(X_local, y_local)
    local_model_stratified   = LinearSVC(**params_local_best_stratified).fit(X_local_stratified, y_local_stratified)
    central_model            = LinearSVC(**params_central_best).fit(X, y)

    analysis.plot_planes(X, y, local_model, local_model_stratified, central_model, distSVM)
