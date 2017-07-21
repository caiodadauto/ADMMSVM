import numpy as np
import pandas as pd
import src.distributedsvm as dist
import src.analysisdata as analysis
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

def plane(distSVM, X, y):

    # Grid Search for distributed SVM
    params_dist_svm = {"C"    : [2],#[2**-15, 2**-10, 2**-5, 2, 2**5, 2**10],
                       "c"    : [10],#[0.5, 1, 10, 20, 100],
                       "max_iter": [250],
                       "step" : [50]
                      }


    # Parameters for SVM
    params_svm = {"C" : [2**-10, 2**-5],#[2**-15, 2**-10, 2**-5, 2, 2**5, 2**10],
                  "max_iter" : [250],
                  "penalty"  : ['l1'],
                  "dual"     : [False]
                 }

    params_dist_best = distSVM.grid_search(X, y, params_dist_svm)
    print(params_dist_best)
    distSVM.set_params(**params_dist_best)
    distSVM.fit(X, y)

    # Prepar data for local and central case
    X_local = pd.read_csv('src/distributedsvm/datas/data_0.csv').values
    y_local = pd.read_csv('src/distributedsvm/datas/class_0.csv').values.T[0]
    scale   = StandardScaler().fit(X_local)
    X_local = scale.transform(X_local)
    scale   = StandardScaler().fit(X)
    X       = scale.transform(X)

    # Grid Search for local and central casa
    gs                  = GridSearchCV(LinearSVC(), params_svm)
    gs.fit(X_local, y_local)
    params_local_best   = gs.best_params_
    print(params_local_best)
    gs.fit(X, y)
    params_central_best = gs.best_params_
    print(params_central_best)

    # Local SVM
    local_model  = LinearSVC(**params_local_best).fit(X_local, y_local)

    # Central SVM
    central_model = LinearSVC(**params_central_best).fit(X, y)

    analysis.plot_planes(X, y, local_model, central_model, distSVM)
