import numpy as np
import pandas as pd
import src.display as display
import src.distributedsvm as dist
import src.analysisdata as analysis
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

nodes, data_info = display.start()
X, y             = analysis.read_data(**data_info)
distSVM          = dist.DistSVM(nodes = nodes)

# Grid Search for distributed SVM
tests = {"$C = 2^{-15}\;\text{e}\;$c = 1"   : {'C': 2**-15, 'c': 1},
         "$C = 2^{-15}\;\text{e}\;$c = 10$" : {'C': 2**-15, 'c': 10},
         "$C = 2^{-10}\;\text{e}\;$c = 1$"  : {'C': 2**-10, 'c': 1},
         "$C = 2^{-10}\;\text{e}\;$c = 10$" : {'C': 2**-10, 'c': 10}
        }

# Parameters for SVM
params = {"C"        : [2**-15, 2**-10, 2**-5, 2],
          "max_iter" : [200]
         }


##
# Cross validation and compare local SVM, central SVM and distributed SVM
##
risk_dist = {"$C = 2^{-15}\;\text{e}\;$c = 1"   : [],
             "$C = 2^{-15}\;\text{e}\;$c = 10$" : [],
             "$C = 2^{-10}\;\text{e}\;$c = 1$"  : [],
             "$C = 2^{-10}\;\text{e}\;$c = 10$" : []
            }
risk_local   = 0
risk_central = 0
skf = StratifiedKFold()
for train_index, test_index in skf.split(X, y):
    for test, params in tests.items():
        distSVM.set_params(**params)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Distributed SVM
        distSVM.fit(X_train, y_train)
        # TODO: same iters...
        iters, risk = distSVM.risk_score(X_test, y_test)
        risk_dist[test].append(risk)
        print(risk_dist)

    # Prepar data to local SVM and central SVM
    X_local_train = pd.read_csv('src/distributedsvm/datas/data_0.csv').values
    y_local_train = pd.read_csv('src/distributedsvm/datas/class_0.csv').values.T[0]
    scale         = StandardScaler().fit(X_local_train)
    X_local_train = scale.transform(X_local_train)
    X_local_test  = scale.transform(X_test)
    scale         = StandardScaler().fit(X_train)
    X_train       = scale.transform(X_train)
    X_test        = scale.transform(X_test)

    # Grid Search for SVM
    gs             = GridSearchCV(LinearSVC(), params)
    gs.fit(X_local_train, y_local_train)
    local_params   = gs.best_params_
    gs.fit(X_train, y_train)
    central_params = gs.best_params_

    # Local SVM
    local_model  = LinearSVC(**local_params).fit(X_local_train, y_local_train)
    risk_local  += 1 - local_model.score(X_local_test, y_test)

    # Central SVM
    central_model = LinearSVC(**central_params).fit(X_train, y_train)
    risk_central  += 1 - central_model.score(X_test, y_test)

for test in risk_dist.keys():
    risk_dist[test] = np.array(risk_dist[test]).sum(axis = 0)/3
print(risk_dist)
risk_local   /= 3
risk_central /= 3
analysis.plot_risk(risk_local, risk_central, risk_dist, iters)
