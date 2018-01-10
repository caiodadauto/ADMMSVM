import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
import pandas as pd
import cvxopt as cvx
from mpi4py import MPI
from pathconf import params_path, datas_path, results_path, graph_path

cvx.solvers.options['show_progress'] = False

comm                 = MPI.COMM_WORLD
data_graph           = pd.read_csv(graph_path.joinpath("graph_mpi.csv"))
index                = data_graph.loc[data_graph['index'].notnull(), 'index'].values
neighborhood         = data_graph['neighborhood'].values
comm_graph           = comm.Create_graph(index, neighborhood)
my_rank              = comm_graph.Get_rank()
my_neighborhood_size = comm_graph.Get_neighbors_count(my_rank)

params               = pd.read_csv(params_path, index_col = [0]).to_dict()['0']
my_file_data         = datas_path.joinpath("data_" + str(my_rank) + ".csv")
my_file_class        = datas_path.joinpath("class_" + str(my_rank) + ".csv")
my_data              = pd.read_csv(my_file_data).values
my_class             = pd.read_csv(my_file_class).values.T[0]

data_size            = my_data.shape[0]
vector_one           = np.ndarray(shape=(data_size, 1), buffer=np.ones(data_size))
my_X                 = np.concatenate((my_data, vector_one), axis=1)
my_Y                 = np.diag(my_class)
dim                  = my_X.shape[1]
my_YX                = np.dot(my_Y, my_X)

c                    = params['c']
C                    = params['C']
my_v                 = np.full(dim, 1)
my_lambda            = np.full(dim, 0)
my_r                 = - 2 * c * my_neighborhood_size * my_v
my_inv_diag_D        = np.concatenate((np.full(dim - 1, 1/(1 + 2 * c * my_neighborhood_size)), np.full(1, 1/(2 * c * my_neighborhood_size))))
my_inv_D             = np.diag(my_inv_diag_D)
my_YX_inv_D          = np.dot(my_YX, my_inv_D)

I                    = np.identity(data_size)
zero                 = np.zeros(data_size)
cte                  = np.full(data_size, C)
my_G                 = cvx.matrix(np.concatenate((-I, I)))
my_h                 = cvx.matrix(np.concatenate((zero, cte)))
my_q                 = cvx.matrix(-(np.ones(data_size) + np.dot(my_YX_inv_D, my_r)))
my_P                 = cvx.matrix(0.5 * np.dot(my_YX_inv_D, my_YX.T))

iterations = range(int(params['max_iter']))
step       = params['step']
for t in iterations:
    my_mu = np.array(cvx.solvers.qp(my_P, my_q, my_G, my_h)['x']).reshape(data_size)

    aux   = np.dot(my_YX.T, my_mu) - my_r
    my_v  = np.dot(my_inv_D, aux)

    my_send_vector       = []
    my_receive_vector    = []
    for neighbor in range(my_neighborhood_size):
        my_send_vector.append(my_v)
        my_receive_vector.append(np.zeros(dim))
    my_send_vector    = np.array(my_send_vector)
    my_receive_vector = np.array(my_receive_vector)

    comm_graph.Neighbor_alltoall(my_send_vector, my_receive_vector)
    receive_sum = np.sum(my_receive_vector, axis = 0)

    my_lambda   = my_lambda + 0.5 * c * (my_neighborhood_size * my_v - receive_sum)

    my_r        = 2 * my_lambda - c * (my_neighborhood_size * my_v + receive_sum)
    my_q        = cvx.matrix(-(np.ones(data_size) + np.dot(my_YX_inv_D, my_r)))

    if (t + 1)%step == 0:
        file = results_path.joinpath("(w,b)_partial_" + str(t + 1) + "_" + str(my_rank) + ".csv")
        pd.DataFrame(my_v).to_csv(file, index = None)
