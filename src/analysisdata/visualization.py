import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler

def visualization(X, y):
    sns.plt.switch_backend('Agg')

    # Normalize the data
    #scale = StandardScaler().fit(X)
    #X     = scale.transform(X)

    # Reduce dimension of the data
    pca = PCA(n_components = 4)
    pca.fit(X)
    X = pca.transform(X)

    # Create data
    data_np = np.concatenate((X, np.array([y]).T), axis=1)
    data    = pd.DataFrame(data=data_np, columns=['pca_1','pca_2','pca_3', 'pca_4', ' '])
    data.loc[data[' '] ==  1, ' '] = 'Com Falha'
    data.loc[data[' '] == -1, ' '] = 'Sem falha'

    # Plot visualization
    g = sns.pairplot(data, hue = ' ', diag_kind = 'kde', palette = 'Set2', markers = ["o", "D"], diag_kws = {'shade': True})
    file = "datas/visualization_variance_(" + str(pca.explained_variance_ratio_.sum()) + ").svg"
    g.savefig(file)
