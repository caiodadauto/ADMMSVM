import numpy as np
import pandas as pd
import networkx as nx

def create_mpi_graph_type(graph):
    index        = []
    neighborhood = []
    sum_degree   = 0

    node_degrees = graph.degree_iter()

    for node, degree in node_degrees:
        sum_degree += degree
        index.append(sum_degree)
        node_neighbors = graph.neighbors_iter(node)
        for neighbor in node_neighbors:
            neighborhood.append(neighbor)

    df = pd.DataFrame({'index': pd.Series(index),
                       'neighborhood':pd.Series(neighborhood)})

    df.to_csv('geometricgraph/graph_mpi.csv', index=False)
