import os
import sys
sys.path.insert(0,'src/')

import tests
import distsvm
import display
import analysisdata as analysis

nodes, data_info = display.start()
dsvm             = distsvm.DSVM(nodes = nodes)
X_set, y_set     = analysis.read_data(**data_info)
X_ran, y_ran     = analysis.create_data()

tests.plane(dsvm, X_ran, y_ran)
# tests.risk(dsvm, X_set, y_set)
