from __future__ import unicode_literals

import numpy as np
import pandas as pd
import seaborn as sns
import  matplotlib.cm as cm
import  matplotlib.pyplot as plt
import  matplotlib.colors as colors
from pathconf import plots_path
def get_average_best_plane(ldsvm):
    n = ldsvm.get_nodes()
    w = []
    b = 0

    for i in range(n):
        plane = ldsvm.get_best_plane(i)
        w.append(plane[0])
        b += plane[1]
    w = np.sum(np.array(w), axis = 0)/n
    b /= n

    return [w, b]

def draw_plane(w, b, color, alpha, line_width, line_style, label = None):
    m  = -w[0]/w[1]
    d  = -b/w[1]
    xx = np.linspace(-4, 4)
    yy = m * xx + d
    plt.plot(xx, yy, lw = line_width, ls=line_style, c = color, label = label, alpha = alpha)

def plot_planes(X, y, local_model, central_model, dist_model):
    sns.set_style('ticks')

    y = pd.DataFrame(y)
    y.loc[y[0] == -1, 0] = colors.to_hex(cm.Pastel2(0))
    y.loc[y[0] == 1, 0]  = colors.to_hex(cm.Pastel2(1))
    y = y.values.T[0]

    w, b = get_average_best_plane(dist_model)

    plt.figure()
    draw_plane(w,
               b,
               cm.tab10(0),
               1,
               2.2,
               '-',
               "SVM Distribuído com C = " + str(dist_model.C) + " e " + "c = " + str(dist_model.c))
    draw_plane(central_model.coef_[0],
               central_model.intercept_[0],
               cm.tab10(1),
               1,
               2.2,
               '--',
               "SVM Central com C = " + str(central_model.get_params()['C']))
    draw_plane(local_model.coef_[0],
               local_model.intercept_[0],
               cm.tab10(2),
               1,
               2.2,
               '-.',
               "SVM Local com C = " + str(local_model.get_params()['C']))
    # draw_plane(local_model_stratified.coef_[0],
    #            local_model_stratified.intercept_[0],
    #            cm.tab10(3),
    #            1,
    #            2.2,
    #            ':',
    #            "SVM Local Estratificado com C = " + str(local_model_stratified.get_params()['C']))


    plt.scatter(X[:, 0], X[:, 1], marker = 'o', c = y, alpha = 0.5)
    sns.despine()
    plt.legend(loc = 2)
    plt.ylim(-4.8, 4.8)
    file = str(plots_path) + "/simple_graph_compare.pdf"
    plt.savefig(file, transparent = True)

    ##
    # Gradiente plot, use max_iter 50 with step 1. Initialized v with (5...5)
    ##
    # plt.figure()
    # palette = sns.color_palette("Blues_d", len(planes))
    # for i in range(len(planes)):
    #     draw_plane(planes[i][0],
    #                planes[i][1],
    #                palette[len(planes) - i - 1],
    #                1,
    #                0.8 + (1.2 * (i + 1)/len(planes)),
    #                '-',
    #                None)
    # plt.scatter(X[:, 0], X[:, 1], marker = 'o', c = y, alpha = 0.5)
    # sns.despine()
    # plt.ylim(-4.8, 4.8)
    # plt.savefig("src/analysisdata/plots/gradient_simple_graph.pdf", transparent = True)

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
    #      plt.ylim(-4.8, 4.8)
    #      plt.savefig(file)
