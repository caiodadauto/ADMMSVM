import numpy as np
import pandas as pd
import src.tests as tests
import src.display as display
import src.distributedsvm as dist
import src.analysisdata as analysis

nodes, data_info = display.start()
distSVM          = dist.DistSVM(nodes = nodes)
#X_set, y_set     = analysis.read_data(**data_info)
X_ran, y_ran     = analysis.create_data()

tests.plane(distSVM, X_ran, y_ran)
#tests.risk(distSVM, X_set, y_set)
