import networkx as nx

def create_connected_geo_graph(nodes):
    if   nodes < 20:
        d = 0.53
    elif nodes >= 20 and nodes < 50:
        d = 0.43
    elif nodes >= 50 and nodes < 100:
        d = 0.23
    elif nodes >= 100 and nodes < 200:
        d = 0.15
    else:
        d = 0.1

    while True:
        graph = nx.random_geometric_graph(nodes,d)
        if nx.is_connected(graph):
            break
    return graph
