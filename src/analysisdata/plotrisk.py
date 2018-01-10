from __future__ import unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathconf import plots_path

def plot_risk(risk_local, risk_central, risk_dist, iters):
    df = pd.DataFrame(risk_dist)
    df.to_csv('graph_risk_test', index=False)
    print('central ', risk_central)
    print('local ', risk_local)
    print('iters ', iters)

    size = iters.shape[0]

    # Create the data
    risk_local   = np.full(size, risk_local)
    risk_central = np.full(size, risk_central)
    iters        = np.array(iters)

    # Plot graphs
    sns.set_style("ticks")
    plt.figure()
    plt.rc('text', usetex = True)
    plt.rc('text.latex', unicode = True)
    tests = list(risk_dist.keys())
    with sns.color_palette("tab20", len(tests) + 2):
        for test in tests:
            label = "SVM distribuído com " + test
            plt.plot(iters, risk_dist[test], linewidth = 2.2, label = label)
        plt.plot(iters, risk_local,   linewidth = 2.2, linestyle = '-.', label = 'SVM Local')
        plt.plot(iters, risk_central, linewidth = 2.2, linestyle = '-.', label = 'SVM Central')
        plt.legend(loc = 'upper right')
        sns.despine()
        plt.xlabel('Iterações')
        plt.ylabel('Risco')
        file = str(plots_path) + "/risk_plot.pdf"
        plt.savefig(file, transparent = True)
