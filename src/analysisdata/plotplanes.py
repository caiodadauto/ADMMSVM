from __future__ import unicode_literals

import numpy as np
import pandas as pd
import seaborn as sns
import  matplotlib.pyplot as plt

def draw_plane(w, b, color, alpha, line_width, label = None):
    m  = -w[0]/w[1]
    d  = -b/w[1]
    xx = np.linspace(-4, 4)
    yy = m * xx + d
    plt.plot(xx, yy, lw = line_width, c = color, label = label, alpha = alpha)

def plot_planes(X, y, local_model, local_model_stratified, central_model, dist_model):
    sns.set_style('darkgrid')

    y = pd.DataFrame(y)
    y.loc[y[0] == -1, 0] = '#FC8D62'
    y.loc[y[0] == 1, 0]  = '#66C2A5'
    y = y.values.T[0]

    planes = dist_model.get_all_planes()

    plt.figure()
    draw_plane(planes[-1][0],
               planes[-1][1],
               "#8DA0CB",
               1,
               1.6,
               "SVM Distribuído")
    draw_plane(central_model.coef_[0],
               central_model.intercept_[0],
               "#FFD92F",
               1,
               1.6,
               "SVM Central")
    draw_plane(local_model.coef_[0],
               local_model.intercept_[0],
               "#E78AC3",
               1,
               1.6,
               "SVM Local")
    draw_plane(local_model_stratified.coef_[0],
               local_model_stratified.intercept_[0],
               "#B3B3B3",
               1,
               1.6,
               "SVM Local Estratificado")


    plt.scatter(X[:, 0], X[:, 1], marker = 'o', c = y, alpha = 0.5)
    plt.legend(loc = 1)
    plt.ylim(-2.3, 2.3)
    plt.savefig("src/analysisdata/plots/simple_graph_compare.svg")

    ##
    # Gradiente plot, use max_iter 50 with step 1. Initialized v with (5...5)
    ##
    # plt.figure()
    # colors = sns.color_palette("hot_d", len(planes))
    # for i in range(len(planes)):
    #     draw_plane(planes[i][0],
    #                planes[i][1],
    #                colors[i],
    #                0.6,
    #                1,
    #                None)
    # plt.scatter(X[:, 0], X[:, 1], marker = 'o', c = y, alpha = 0.5)
    # plt.ylim(-2.3, 2.3)
    # plt.savefig("src/analysisdata/plots/gradient_simple_graph.svg")

    # colors = sns.color_palette("hot_d", len(planes))
    # for i in range(len(planes)):
    #      plt.figure()
    #      legend = "Iteração " + str(i + 1)
    #      draw_plane(planes[i][0],
    #                 planes[i][1],
    #                 colors[i],
    #                 1,
    #                 1.6,
    #                 legend)
    #      plt.scatter(X[:, 0], X[:, 1], marker = 'o', c = y, alpha = 0.5)
    #      file = "src/analysisdata/plots/gif/" + str(i) + ".svg"
    #      plt.legend(loc = 2)
    #      plt.ylim(-2.3, 2.3)
    #      plt.savefig(file)
