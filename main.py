import subprocess as sub
import src.display as display
import src.analysisdata as analy
import src.connectedgraph as c_graph
from sklearn.model_selection import StratifiedKFold

nodes, data_info = display.start()

##
# Set environment
##
G = c_graph.create_connected_geo_graph(nodes)

c_graph.create_mpi_graph_type(G)

c_graph.show_geo_graph(G)

X, y = analy.read_data(**data_info)

##
# Cross validation
##
skf = StratifiedKFold()
command = "mpiexec -n " + str(nodes) + " python mpi.py"
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    analy.nodes_split_data(X_train, y_train, nodes)

    sub.check_call(command, shell = True)

sub.check_call('rm datasfornode/*.csv', shell = True)
