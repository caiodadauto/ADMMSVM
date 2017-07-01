import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_risk(risk_local, risk_central, risk_dist, iters):
    sns.set_style("darkgrid")
    size = iters.shape[0]
    f    = plt.figure()
    '''
    aux  = np.concatenate((risk_dist,[risk_local],[risk_central]))
    ymax = np.max(aux)
    ymin = np.min(aux)
    '''


    # Create the data
    risk_local   = np.full(size, risk_local)
    risk_central = np.full(size, risk_central)
    iters        = np.array(iters)

    # Plot graphs
    tests = list(risk_dist.keys())
    with sns.color_palette("muted", len(tests)):
        for test in tests:
            plt.plot(iters, risk_dist[test], linewidth = 2, label = test)
        plt.plot(iters, risk_local,   linewidth = 2, linestyle = '--', color = 'r', label = 'Local SVM')
        plt.plot(iters, risk_central, linewidth = 2, linestyle = '--', color = 'g', label = 'Central SVM')
        plt.legend(loc = 'upper right')
        plt.show()
