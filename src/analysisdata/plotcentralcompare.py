from __future__ import unicode_literals

import numpy as np
import seaborn as sns
import  matplotlib.cm as cm
import  matplotlib.pyplot as plt
import  matplotlib.colors as colors
from pathconf import plots_path

def plot(delta_iters, iters):
    sns.set_style('ticks')
    plt.plot(iters, delta_iters, linewidth = 2, c = cm.tab20(0))
    sns.despine()
    plt.ylim(delta_iters.min()*(0.6), delta_iters.max()*(1.4))

    plt.xlabel("Iterações " + r'$(l)$')
    plt.ylabel(r'$\Delta^l(v_c)$')
    file = str(plots_path) + "/dispersion_central_graph.pdf"
    plt.savefig(file, transparent = True)

def metric(vectors, cv, n):
    delta = 0
    for i in range(n):
        delta += np.linalg.norm(vectors[i] - cv)

    return delta/n

def plot_central_compare(dsvm, csvm):
    n               = dsvm.get_nodes()
    iters           = dsvm.get_iters()
    planes_per_node = dsvm.get_all_planes()

    for i in range(n):
        for j in range(len(iters)):
            v       = np.array(planes_per_node[i][j][0])
            norm_w  = np.linalg.norm(v)
            v       = np.insert(v, -1, planes_per_node[i][j][1])
            v      /= norm_w
            planes_per_node[i][j] = v
    planes_per_node = np.array(planes_per_node)
    planes_per_iter = np.transpose(planes_per_node, axes=[1,0,2])

    norm_w   = np.linalg.norm(np.array(csvm.coef_[0]))
    cvector  = np.array([csvm.coef_[0][0], csvm.coef_[0][1], csvm.intercept_[0]])
    cvector /= norm_w

    delta_iters = []
    for i in range(len(iters)):
        vectors = planes_per_iter[i]
        delta_iters.append(metric(vectors, cvector, n))

    plt.figure()
    plot(np.array(delta_iters), iters)
