import numpy as np
import pandas as pd

def create_chess():
    delta = 3
    size  = 4
    seed  = 2
    nodes = int(size * size / 2)

    rand = np.random.RandomState()
    rand.seed(seed)

    n_coor = 2560
    v = rand.uniform(size = n_coor)
    delta_coor = int(n_coor / (size * size))
    n_points = int(delta_coor/2)

    chess_data = []
    chess_class = []
    coor_square = 0
    square_class = -1
    y_start = -size/2 * delta
    for i in range(size):
        x_start = -size/2 * delta
        for j in  range(size):
            square_class *= -1
            square_coor = v[:delta_coor]
            v = np.delete(v,np.s_[:delta_coor])
            start = [x_start, y_start]
            for k in range(n_points):
                point = []
                for l in range(2):
                    point.append(square_coor[k * 2 + l] * delta + start[l])
                chess_data.append(point)
                chess_class.append(square_class)
            x_start += delta
        y_start += delta
        square_class *= -1

    X = np.array(chess_data)
    y = np.array(chess_class)

    return [X, y]
