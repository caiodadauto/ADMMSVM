import numpy as np
import pandas as pd
import seaborn as sns
import  matplotlib.pyplot as plt
from pathconf import plots_path


def make_meshgrid(x, y, h = .02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def predict(dsvm, node, mesh):
    alpha, beta, b = ndsvm.get_classifier(node)

    p = []
    for v in np.nditer(mesh):
        p.append(ndsvm.local_discriminant(node, alpha, beta, b, v))

    return np.array(p)

def plot_contours_scatter(ndsvm, node, xx, yy, X, y, **params):
    plt.figure()

    print("Doing predict")
    z = predict(ndsvm, node, np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, **params)
    plt.scatter(X, y, c = y, cmap = plt.cm.coolwarm, s=25, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    file = str(plots_path) + "/non_linear_classifier.pdf"
    plt.savefig(file, transparent = True)


def plot_non_linear_classifier(ndsvm, node, X, y):
    sns.set_style('ticks')

    xx, yy = make_meshgrid(X[:, 0], X[:, 1])

    plot_contours_scatter(ndsvm, node, xx, yy, X, y, cmap=plt.cm.coolwarm, alpha=0.5)
