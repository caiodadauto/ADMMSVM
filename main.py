import os
import sys
sys.path.insert(0,'src/')

import tests
import distsvm
import display
import analysisdata as analysis

nodes, linear_or_not, data_info = display.start()
X_set, y_set = analysis.read_data(**data_info)
if linear_or_not is "linear":
    ldsvm        = distsvm.LDSVM(nodes = nodes)
    X_ran, y_ran = analysis.create_data()
    tests.artificial(ldsvm, X_ran, y_ran)
    tests.real(ldsvm, X_set, y_set)
else:
    ndsvm = distsvm.NDSVM(nodes = nodes)
    tests.chess(ndsvm, X_set, y_set)
