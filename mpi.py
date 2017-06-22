import numpy as np
import pandas as pd
from mpi4py import MPI

#Communicator for all process
comm = MPI.COMM_WORLD

#Get topology from csv file
data_graph   = pd.read_csv('geometricgraph/graph_mpi.csv')
index        = data_graph.loc[data_graph['index'].notnull(), 'index'].values
neighborhood = data_graph['neighborhood'].values

#Create a new communicator with a topology from a predefined graph
comm_graph = comm.Create_graph(index, neighborhood)

my_rank = comm_graph.Get_rank()

my_neighborhood_size = comm_graph.Get_neighbors_count(my_rank)

my_neighborhood_rank = np.full([my_neighborhood_size], -1)

comm_graph.Neighbor_alltoall(np.full([my_neighborhood_size],my_rank), my_neighborhood_rank)

print("Process with rank ", my_rank, " rank of neighborhood ", my_neighborhood_rank)
