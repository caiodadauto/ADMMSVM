import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def show_geo_graph(graph):
    node_position = nx.get_node_attributes(graph,'pos')

    # find node near center (0.5,0.5)
    d_min            = 1
    node_near_center = 0
    for node in node_position:
        x, y = node_position[node]
        distance = (x - 0.5)**2 + (y - 0.5)**2
        if distance < d_min:
            node_near_center = node
            d_min = distance

    # color by path length from node near center
    color_node        = dict(nx.single_source_shortest_path_length(graph, node_near_center))
    array_color_node  = np.array(list(color_node.values()))

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(graph, node_position, nodelist=[node_near_center],alpha=0.4)
    nx.draw_networkx_nodes(graph, node_position, nodelist=color_node.keys(),
                           node_size = 80,
                           node_color = array_color_node,
                           cmap = plt.cm.winter_r)

    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    plt.axis('off')
    plt.savefig('geometricgraph/random_geometric_graph.svg')
