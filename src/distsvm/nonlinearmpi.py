import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
import pandas as pd
import cvxopt as cvx
from mpi4py import MPI
from kernels import rbf
from scipy import linalg
from pathconf import params_path, datas_path, results_path, graph_path


def K(X, Y, gamma):
    K = []
    X_n_data = X.shape[0]
    Y_n_data = Y.shape[0]

    for i in range(X_n_data):
        line = []
        for j in range(Y_n_data):
            line.append(rbf(X[i], Y[j], gamma))
        K.append(line)
    return np.array(K)

def cho_solve_matrix(cholesk_factor, X):
    n_col = X.shape[1]

    result = []
    for j in range(n_col):
        result.append(linalg.cho_solve(cholesk_factor, X[:,j]))
    return np.array(result).T

def main():
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
    my_class             = pd.read_csv(my_file_class).values.T[0]
    my_X                 = pd.read_csv(my_file_data).values
    my_Y                 = np.diag(my_class)
    data_size            = my_X.shape[0]
    chi                  = pd.read_csv(datas_path.joinpath("chi.csv")).values
    dim                  = chi.shape[0]

    gamma                = params['gamma']
    c                    = params['c']
    C                    = params['C']

    K_X                  = K(my_X, my_X, gamma)
    K_chi                = K(chi, chi, gamma)
    K_Xchi               = K(my_X, chi, gamma)

    network_cte          = 2 * c * my_neighborhood_size
    Q                    = np.identity(dim) + network_cte * K_chi
    cholesk_Q            = linalg.cho_factor(Q)
    inv_QK_chiX          = cho_solve_matrix(cholesk_Q, K_Xchi.T)
    inv_QK_chi           = cho_solve_matrix(cholesk_Q, K_chi)
    tilde_K_X            = network_cte * np.dot(K_Xchi, inv_QK_chiX)
    tilde_K_chi          = network_cte * np.dot(K_chi, inv_QK_chi)
    tilde_K_Xchi         = network_cte * np.dot(K_Xchi, inv_QK_chi)

    my_tilde_omega       = np.full(dim, 1)
    my_lambda            = np.full(dim, 0)
    my_b                 = 1.
    my_zeta              = 0.
    my_r                 = - network_cte * my_tilde_omega
    my_s                 = - network_cte * my_b

    I                    = np.identity(data_size)
    zero                 = np.zeros(data_size)
    cte                  = np.full(data_size, C)
    one                  = np.ones(data_size)
    aux_P                = 1 / network_cte * np.ones((data_size, data_size))
    aux_r_q              = np.dot(my_Y, (K_Xchi - tilde_K_Xchi))
    aux_s_q              = 1 / network_cte * np.dot(my_Y, one)
    my_G                 = cvx.matrix(np.concatenate((-I, I)))
    my_h                 = cvx.matrix(np.concatenate((zero, cte)))
    my_q                 = cvx.matrix(-(one + np.dot(aux_r_q, my_r) + my_s * aux_s_q))
    my_P                 = cvx.matrix(np.dot(my_Y, np.dot((K_X - tilde_K_X + aux_P), my_Y)))

    iterations = range(int(params['max_iter']))
    for t in iterations:
        print("Iteration ", t, " rank ", my_rank)
        my_mu = np.array(cvx.solvers.qp(my_P, my_q, my_G, my_h)['x']).reshape(data_size)

        aux_mu_omega   = np.dot((K_Xchi.T - tilde_K_Xchi.T), my_Y)
        my_tilde_omega = np.dot(aux_mu_omega, my_mu) - np.dot((K_chi - tilde_K_chi), my_r)
        aux_b          = np.dot(my_mu, np.dot(my_Y, one)) - my_s
        my_b           = 1 / network_cte * aux_b

        my_send_matrix       = []
        my_receive_matrix    = []
        my_send_vector       = np.concatenate((my_tilde_omega, np.array([my_b])))
        for neighbor in range(my_neighborhood_size):
            my_send_matrix.append(my_send_vector)
            my_receive_matrix.append(np.zeros(dim + 1))
        my_send_matrix    = np.array(my_send_matrix)
        my_receive_matrix = np.array(my_receive_matrix)
        comm_graph.Neighbor_alltoall(my_send_matrix, my_receive_matrix)
        receive_sum     = np.sum(my_receive_matrix, axis = 0)
        tilde_omega_sum = receive_sum[:dim]
        b_sum           = receive_sum[-1]

        my_lambda = my_lambda + 0.5 * c * (my_neighborhood_size * my_tilde_omega - tilde_omega_sum)
        my_zeta   = my_zeta + 0.5 * c * (my_neighborhood_size * my_b - b_sum)

        my_r      = 2 * my_lambda - c * (my_neighborhood_size * my_tilde_omega + tilde_omega_sum)
        my_s      = 2 * my_zeta - c * (my_neighborhood_size * my_b + b_sum)

        my_q      = cvx.matrix(-(one + np.dot(aux_r_q, my_r) + my_s * aux_s_q))

    my_alpha   = np.dot(my_Y, my_mu)
    my_beta    = network_cte * (np.dot(inv_QK_chi, my_r) - np.dot(inv_QK_chiX, my_alpha)) - my_r

    file_alpha = results_path.joinpath("alpha_" + str(my_rank) + ".csv")
    file_beta  = results_path.joinpath("beta_" + str(my_rank) + ".csv")
    file_b     = results_path.joinpath("b_" + str(my_rank) + ".csv")
    pd.DataFrame(my_alpha).to_csv(file_alpha, index = None)
    pd.DataFrame(my_beta).to_csv(file_beta, index = None)
    pd.DataFrame([my_b]).to_csv(file_b, index = None)

if __name__ == "__main__":
    main()
