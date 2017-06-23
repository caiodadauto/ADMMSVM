import cvxopt
import numpy as np
import pandas as pd
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
my_v      = np.full(dim, my_rank)
my_lambda = np.full(dim, 0)

# Initializes parameters of method
c = 1
C = 60

for t in range(100):
    # Create the send and receive vectors and communicate with neighbors
    my_send_vector       = []
    my_receive_vector    = []

    my_neighborhood_size = comm_graph.Get_neighbors_count(my_rank)

    for neighbor in range(my_neighborhood_size):
        my_send_vector.append(my_v)
        my_receive_vector.append(np.full(dim, 0))

    my_send_vector       = np.array(my_send_vector)
    my_receive_vector    = np.array(my_receive_vector)

    # Change information about v amoung my neighborhood
    comm_graph.Neighbor_alltoall(my_send_vector, my_receive_vector)

    # Start the algorithm
    
