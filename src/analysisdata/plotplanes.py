from __future__ import unicode_literals

import numpy as np
import pandas as pd
import seaborn as sns
import  matplotlib.pyplot as plt

def plot_planes(X, y, local_model, central_model, dist_model):
    xx       = np.linspace(-4, 4)

    y = pd.DataFrame(y)
    y.loc[y[0] == -1, 0] = '#FC8D62'
    y.loc[y[0] == 1, 0]  = 'gray'
    y = y.values.T[0]

    planes = dist_model.get_all_planes()

    '''
    sns.set_style('darkgrid')
    for i in range(len(planes)):
        plt.figure()
        a  = - planes[i][0][0]
        b  = planes[i][1]
        yy = a * xx + b
        legend = "Iteração " + str(i + 1)
        plt.plot(xx, yy, c = "#3498db", lw = 1.6, label = legend)
        plt.scatter(X[:, 0], X[:, 1], marker = 'o', c = y, alpha = 0.5)
        file = "gif/" + str(i) + ".pdf"
        plt.legend(loc = 1)
        plt.ylim(-2.3, 2.3)
        plt.savefig(file)
    '''
    '''
    sns.set_style('darkgrid')
    plt.figure()
    with sns.color_palette("GnBu_d", 50):
        for i in range(len(planes) - 1, -1, -1):
            a  = - planes[i][0][0]
            b  = planes[i][1]
            yy = a * xx + b
            plt.plot(xx, yy, lw = 0.7, alpha = 0.5)
    plt.scatter(X[:, 0], X[:, 1], marker = 'o', c = y, alpha = 0.5)
    plt.ylim(-2.3, 2.3)
    plt.savefig("out.svg")
    '''
    plt.figure()
    print(len(planes))
    w  = planes[len(planes) - 1][0]
    a  = -w[0]/w[1]
    b  = -planes[len(planes) - 1][1]/w[1]
    yy = a * xx + b
    print(a)
    print(b)
    plt.plot(xx, yy, lw = 1.5, alpha = 0.5, c = "#37535E", label = "Distributed")

    w  = local_model.coef_[0]
    a  = -w[0]/w[1]
    b  = -local_model.intercept_[0]/w[1]
    yy = a * xx + b
    print(a)
    print(b)
    plt.plot(xx, yy, lw = 1.5, alpha = 0.5, c = "#A6D854", label = "Local")

    w  = central_model.coef_[0]
    a  = -w[0]/w[1]
    b  = -central_model.intercept_[0]/w[1]
    yy = a * xx + b
    print(a)
    print(b)
    plt.plot(xx, yy, lw = 1.5, alpha = 0.5, c = "#FFD92F", label = "Central")

    plt.scatter(X[:, 0], X[:, 1], marker = 'o', c = y, alpha = 0.5)
    plt.legend(loc = 1)
    #plt.ylim(-2.3, 2.3)
    plt.savefig("out1.svg")
