import numpy as np

def rbf(x, y, gamma):
    return np.exp(-gamma * np.dot(x - y, x - y))
