# coding: utf-8
import numpy as np
import pandas as pd

delta = 2
size = 6
nodes = int(size * size / 2)

rand = np.random.RandomState()
rand.seed(2)

n_coor = 7200
v = rand.uniform(size=n_coor)
delta_coor = int(n_coor / (size * size))
n_points = int(delta_coor/2)

chess_data = []
chess_class = []
coor_square = 0
square_class = -1
y_start = -6
for i in range(size):
    x_start = -6
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
frame_X = pd.DataFrame(X)
frame_y = pd.DataFrame(y)
frame_X.to_csv('chess/chess_data.csv', index = False)
frame_y.to_csv('chess/chess_class.csv', index = False)

for node in range(nodes):
    start_slice = node * 2 * n_points
    final_slice = start_slice + 2 * n_points
    dx = X[start_slice:final_slice]
    dy = y[start_slice:final_slice]

    frame_dx = pd.DataFrame(dx)
    frame_dy = pd.DataFrame(dy)
    file_data = 'chess/bad_chess/data_' + str(node) + '.csv'
    file_class = 'chess/bad_chess/class_' + str(node) + '.csv'
    frame_dx.to_csv(file_data, index = False)
    frame_dy.to_csv(file_class, index = False)

