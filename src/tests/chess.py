import numpy as np
import pandas as pd
import analysisdata as analysis
from pathconf import datas_path, results_path, graph_path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

def chess(ndsvm, X, y):
    params_dist_svm = {
            "C"        : 1,#
            "c"        : 2,#
            "gamma"    : 0.7,
            "p"        : 1200,
            "max_iter" : 3000
            }


    params_svm = {
            "C"        : 1,#
            "gamma"    : 0.7,
            "max_iter" : 1000
            }

    scaler  = StandardScaler()
    ndsvm.network.split_data(X, y, stratified=False)
    local_data    = str(datas_path) + "/data_0.csv"
    local_class   = str(datas_path) + "/class_0.csv"
    X_local       = pd.read_csv(local_data).values
    y_local       = pd.read_csv(local_class).values.T[0]
    X_local_scale = scaler.fit_transform(X_local)
    X_scale = scaler.fit_transform(X)

    local_model   = SVC(**params_svm).fit(X_local_scale, y_local)
    central_model = SVC(**params_svm).fit(X_scale, y)
    ndsvm.set_params(**params_dist_svm)
    ndsvm.fit(X_scale, y)

    analysis.plot_non_linear_classifier(ndsvm, central_model, local_model, 0, X, y)
