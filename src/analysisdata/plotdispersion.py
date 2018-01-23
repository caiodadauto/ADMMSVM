from __future__ import unicode_literals

import numpy as np
import seaborn as sns
import  matplotlib.cm as cm
import  matplotlib.pyplot as plt
import  matplotlib.colors as colors
from pathconf import plots_path

def plot(delta_iters, iters):
    sns.set_style('ticks')
    plt.plot(iters, delta_iters, linewidth = 2.2, c = cm.tab20(3))
    sns.despine()
    plt.ylim(delta_iters.min()*(0.5), delta_iters.max()*(1.4))

    plt.xlabel("Iterações " + r'(l)')
    plt.ylabel(r'$\Delta^l(\bar{v})$')
    file = str(plots_path) + "/dispersion_graph.pdf"
    plt.savefig(file, transparent = True)

def metric(vectors, mean, n):
    delta = 0
    for i in range(n):
        delta += np.linalg.norm(vectors[i] - mean)

    return delta/n

def plot_dispersion(ldsvm):
    n               = ldsvm.get_nodes()
    iters           = ldsvm.get_iters()
    planes_per_node = ldsvm.get_all_planes()

    for i in range(n):
        for j in range(len(iters)):
            v = np.array(planes_per_node[i][j][0])
            v = np.insert(v, -1, planes_per_node[i][j][1])
            planes_per_node[i][j] = v
    planes_per_node = np.array(planes_per_node)
    planes_per_iter = np.transpose(planes_per_node, axes=[1,0,2])

    delta_iters = []
    for i in range(len(iters)):
        vectors = planes_per_iter[i]
        delta_iters.append(metric(vectors, vectors.sum(axis = 0)/n, n))

    plt.figure()
    plot(np.array(delta_iters), iters)
