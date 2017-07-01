import numpy as np
import pandas as pd
import cvxopt as cvx
from mpi4py import MPI

cvx.solvers.options['show_progress'] = False

# Communicator for all process
comm = MPI.COMM_WORLD

# Read Params
params = pd.read_csv('src/distributedsvm/params.csv', index_col = [0]).to_dict()['0']

# Get topology from csv file
data_graph   = pd.read_csv('src/distributedsvm/graph/graph_mpi.csv')
index        = data_graph.loc[data_graph['index'].notnull(), 'index'].values
neighborhood = data_graph['neighborhood'].values

# Create a new communicator with a topology from a predefined graph and get information about that topology
comm_graph           = comm.Create_graph(index, neighborhood)
my_rank              = comm_graph.Get_rank()
my_neighborhood_size = comm_graph.Get_neighbors_count(my_rank)

# Get data set of that pprocessor
my_file_data  = "src/distributedsvm/datas/data_" + str(my_rank) + ".csv"
my_file_class = "src/distributedsvm/datas/class_" + str(my_rank) + ".csv"
my_data       = pd.read_csv(my_file_data).values
my_class      = pd.read_csv(my_file_class).values.T[0]

# Configure matrix X and Y for the algorithm SVM-ADMM
data_size  = my_data.shape[0]
vector_one = np.ndarray(shape=(data_size, 1), buffer=np.ones(data_size))
my_X       = np.concatenate((my_data, vector_one), axis=1)
my_Y       = np.diag(my_class)
dim        = my_X.shape[1]
my_YX      = np.dot(my_Y, my_X)

# Initializes parameters of method
c             = params['c']
C             = params['C']
if my_rank == 0:
    print("C: ", C,  "c: ", c)
my_v          = np.full(dim, 1)
my_lambda     = np.full(dim, 0)
my_r          = - c * my_neighborhood_size * my_v
my_inv_diag_D = np.concatenate((np.full(dim - 1, 1/(1 + c * my_neighborhood_size)), np.full(1, 1/(c * my_neighborhood_size))))
my_inv_D      = np.diag(my_inv_diag_D)
my_YX_inv_D   = np.dot(my_YX, my_inv_D)

# Initializes parameters of cvxopt package
I    = np.identity(data_size)
zero = np.zeros(data_size)
cte  = np.full(data_size, C)
my_G = cvx.matrix(np.concatenate((-I, I)))
my_h = cvx.matrix(np.concatenate((zero, cte)))
my_q = cvx.matrix(-(np.ones(data_size) + np.dot(my_YX_inv_D, my_r)))
my_P = cvx.matrix(2 * np.dot(my_YX_inv_D, np.dot(my_X.T, my_Y)))


# Start the algorithm
for t in range(params['max_iter']):
    # Figure out mu
    my_mu = np.array(cvx.solvers.qp(my_P, my_q, my_G, my_h)['x']).reshape(data_size)

    # Figure out v
    aux = np.dot(my_YX.T, my_mu) - my_r
    my_v = np.dot(my_inv_D, aux)

    # Create send and receive vector
    my_send_vector       = []
    my_receive_vector    = []
    for neighbor in range(my_neighborhood_size):
        my_send_vector.append(my_v)
        my_receive_vector.append(np.zeros(dim))
    my_send_vector    = np.array(my_send_vector)
    my_receive_vector = np.array(my_receive_vector)

    # Change information about v amoung the neighborhood with sincronization
    comm_graph.Neighbor_alltoall(my_send_vector, my_receive_vector)
    receive_sum = np.sum(my_receive_vector, axis = 0)

    #Figure out lambda
    my_lambda = my_lambda + 0.5 * c * (my_neighborhood_size * my_v - receive_sum)

    # Figure out the parameter r from method and parameter q from cvxopt
    my_r = my_lambda - 0.5 * c * (my_neighborhood_size * my_v + receive_sum)
    my_q = cvx.matrix(-(np.ones(data_size) + np.dot(my_YX_inv_D, my_r)))

    # Get information for analysis
    if my_rank == 0 and (t + 1)%5 == 0:
        name = "src/distributedsvm/results/(w,b)_partial_" + str(t + 1) + ".csv"
        pd.DataFrame(my_v).to_csv(name, index = None)
