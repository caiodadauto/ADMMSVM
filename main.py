import os
import sys
sys.path.insert(0,'src/')

import tests
import distsvm
import display
import analysisdata as analysis

nodes, test_info = display.start()
X, y = analysis.read_data(**test_info['data_info'])
if test_info['type'] == 'linear':
    ldsvm = distsvm.LDSVM(nodes = nodes)
    if test_info['name'] == 'artificial':
        tests.artificial(ldsvm, X, y)
    else:
        tests.real(ldsvm, X, y)
else:
    ndsvm = distsvm.NDSVM(nodes = nodes)
    if test_info['name'] == 'artificial':
        tests.chess(ndsvm, X, y)
    else:
        tests.cancer(ndsvm, X, y)
