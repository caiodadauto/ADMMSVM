import numpy as np
from sklearn.datasets import make_classification

def create_data():
    X, y = make_classification(n_samples            = 2500,
                               n_features           = 2,
                               n_redundant          = 0,
                               n_informative        = 2,
                               n_clusters_per_class = 1,
                               random_state         = 7)
    np.place(y, y == 0, [-1])

    return [X, y]
