import numpy as np
from sklearn.datasets import make_circles

def create_circles():
    X, y = make_circles(n_samples = 1000, noise = 0.075)
    np.place(y, y == 0, [-1])

    return [X, y]
