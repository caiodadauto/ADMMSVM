import numpy as np
import pandas as pd
import cvxopt as cvx
from mpi4py import MPI

# Communicator for all process
comm = MPI.COMM_WORLD

# Get topology from csv file
data_graph   = pd.read_csv('geometricgraph/graph_mpi.csv')
index        = data_graph.loc[data_graph['index'].notnull(), 'index'].values
neighborhood = data_graph['neighborhood'].values

# Create a new communicator with a topology from a predefined graph
comm_graph = comm.Create_graph(index, neighborhood)

my_rank = comm_graph.Get_rank()

# Get my set of data
my_file_data  = "datasfornode/data_" + str(my_rank) + ".csv"
my_file_class = "datasfornode/class_" + str(my_rank) + ".csv"
my_data  = pd.read_csv(my_file_data).values
my_class = pd.read_csv(my_file_class).values.T[0]

# Configure matrix X and Y from the algorithm SVM-ADMM
n_rows     = my_data.shape[0]
vector_one = np.ndarray(shape=(n_rows, 1), buffer=np.ones(n_rows))
my_X       = np.concatenate((my_data, vector_one), axis=1)
my_Y       = np.diag(my_class)

print("Processo ", my_rank, "dimensao de X ", my_X.shape, "dimensao de y ", my_Y.shape)
# Initializes the vectors v and lambda
dim       = my_X.shape[1]
my_v      = np.full(dim, 1)
my_lambda = np.full(dim, 0)

# Create the send and receive vectors and communicate with neighbors
my_send_vector       = []
my_receive_vector    = []

my_neighborhood_size = comm_graph.Get_neighbors_count(my_rank)

for neighbor in range(my_neighborhood_size):
    my_send_vector.append(my_v)
    my_receive_vector.append(my_v)

my_send_vector       = np.array(my_send_vector)
my_receive_vector    = np.array(my_receive_vector)

# Initializes parameters of method
c    = 1
C    = 60

aux           = c * my_neighborhood_size
my_inv_diag_D = np.concatenate((np.full(dim - 1, 1/(1 + aux)), np.full(1, 1/aux)))
my_inv_D      = np.diag(my_inv_diag_D)

my_YX_inv_D   = np.dot(np.dot(my_Y, my_X), my_inv_D)

# Initializes parameters of cvxopt package
I    = np.identity(n_rows)
my_G = cvx.matrix(np.concatenate((-I, I)))

zero = np.zeros(n_rows)
cte  = np.full(n_rows, C)
my_h = cvx.matrix(np.concatenate((zero, cte)))

P = cvx.matrix(2 * np.dot(my_YX_inv_D, np.dot(X.T, Y)))

# Start the algorithm
for t in range(100):
    # Figure out the sum of receive vectors
    receive_sum = np.sum(my_receive_vector, axis = 0)

    # Initializes parameter r from method and parameter q from cvxopt
    my_r = my_lambda - 0.5 * c * (my_neighborhood_size * my_v + receive_sum)
    q    = cvx.matrix(-(np.ones(n_rows) + np.dot(my_YX_inv_D)))

    # Figure out mu
    my_mu = cvx.solvers.qp(P, q, G, h)['x']

    # Figure out v
    my_v = np.dot(my_inv_D, np.dot(my_X, np.dot(my_Y, my_mu)) - my_lambda + 0.5 * c * (my_neighborhood_size * my_v + receive_sum))

    # Change information about v amoung my neighborhood
    comm_graph.Neighbor_alltoall(my_send_vector, my_receive_vector)

    # Figure out the sum of receive vectors
    receive_sum = np.sum(my_receive_vector, axis = 0)

    #Figure out lambda
    my_lambda = my_lambda + 0.5 * c * (my_neighborhood_size * my_v - receive_sum)

