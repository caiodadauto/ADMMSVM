import numpy as np
import subprocess as sub
import src.display as display
import src.analysisdata as analy
import src.connectedgraph as c_graph
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

nodes, data_info = display.start()

##
# Set environment
##
G = c_graph.create_connected_geo_graph(nodes)

c_graph.create_mpi_graph_type(G)

c_graph.show_geo_graph(G)

X, y = analy.read_data(**data_info)

##
# Cross validation and compare local SVM, central SVM and distributed SVM
##
acc_dist    = []
acc_local   = 0
acc_central = 0
skf = StratifiedKFold()
command = "mpiexec -n " + str(nodes) + " python mpi.py"
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    analy.nodes_split_data(X_train, y_train, nodes)

    # Distributed SVM
    sub.check_call(command, shell = True)
    iter, acc = analy.accuracy_score(X_test, y_test)
    acc_dist.append(acc)

    # Local SVM
    X_local_train, y_local_train = analy.get_local_data()
    local_model                  = SVC(C = 60, kernel = 'linear', max_iter = 200).fit(X_local_train, y_local_train)
    acc_local                   += local_model.score(X_test, y_test)

    # Central SVM
    central_model = SVC(C = 60, kernel = 'linear', max_iter = 200).fit(X_train, y_train)
    acc_central  += central_model.score(X_test, y_test)
acc_dist     = np.array(acc_dist).sum(axis = 0)/3
acc_local   /= 3
acc_central /= 3
print("dist ", acc_dist)
print("local ", acc_local)
print("central ", acc_central)
#analy.plot_accuracy(acc_local/3, acc_central/3)

# Clear directories
sub.check_call('rm partialresults/*.csv', shell = True)
sub.check_call('rm datasfornode/*.csv', shell = True)
