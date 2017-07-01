import numpy as np
import pandas as pd
import src.display as display
import src.distributedsvm as dist
import src.analysisdata as analysis
from sklearn.svm import SVC
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

nodes, data_info = display.start()
X, y             = analysis.read_data(**data_info)
distSVM          = dist.DistSVM(nodes = nodes)

# TODO: Grid Search?
tests = {'C = 20' : {'C': 20,
                    'c': 1},
         'C = 60' : {'C': 60,
                    'c': 100},
         'C = 200': {'C': 200,
                    'c': 10}
        }

##
# Cross validation and compare local SVM, central SVM and distributed SVM
##
risk_dist    = {'C = 20' : [],
                'C = 60' : [],
                'C = 200': [],
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

    # Local SVM
    local_model  = SVC(C = 60, kernel = 'linear').fit(X_local_train, y_local_train)
    risk_local  += 1 - local_model.score(X_local_test, y_test)

    # Central SVM
    central_model = SVC(C = 60, kernel = 'linear').fit(X_train, y_train)
    risk_central  += 1 - central_model.score(X_test, y_test)

for test in risk_dist.keys():
    risk_dist[test] = np.array(risk_dist[test]).sum(axis = 0)/3
print(risk_dist)
risk_local   /= 3
risk_central /= 3
analysis.plot_risk(risk_local, risk_central, risk_dist, iters)
