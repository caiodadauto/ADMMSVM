from __future__ import unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_risk(risk_local, risk_central, risk_dist, iters):
    size = iters.shape[0]

    # Create the data
    risk_local   = np.full(size, risk_local)
    risk_central = np.full(size, risk_central)
    iters        = np.array(iters)

    # Plot graphs
    sns.set_style("darkgrid")
    plt.figure()
    plt.rc('text', usetex = True)
    plt.rc('text.latex', unicode = True)
    tests = list(risk_dist.keys())
    with sns.color_palette("Set2", len(tests) + 2):
        for test in tests:
            label = "SVM distribuído com " + test
            plt.plot(iters, risk_dist[test], linewidth = 1.7, label = label)
        plt.plot(iters, risk_local,   linewidth = 1.7, linestyle = '--', label = 'SVM Local')
        plt.plot(iters, risk_central, linewidth = 1.7, linestyle = '--', label = 'SVM Central')
        plt.legend(loc = 'upper right')
        plt.xlabel('Iterações')
        plt.ylabel('Risco')
        plt.savefig('src/analysisdata/plots/risk_plot.svg')
