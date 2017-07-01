import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import subprocess as sub
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

class Network(object):
    def __init__(self, nodes):
        if type(nodes) == str:
            # TODO
            print('Under construction!! Nodes = 10')
            exit()
        else:
            self.nodes = nodes
            self.graph = self.connected_geo_graph(self.nodes)
            self.show_geo_graph(self.graph, 'src/distributedsvm/graph')

    def create_graph_mpi(self, file):
        index        = []
        neighborhood = []
        sum_degree   = 0

        node_degrees = self.graph.degree_iter()

        for node, degree in node_degrees:
            sum_degree += degree
            index.append(sum_degree)
            node_neighbors = self.graph.neighbors_iter(node)
            for neighbor in node_neighbors:
                neighborhood.append(neighbor)

        df = pd.DataFrame({'index': pd.Series(index),
                           'neighborhood':pd.Series(neighborhood)})

        df.to_csv(file, index=False)

    def split_data(self, X, y, data_dir):
        node = 0
        skf  = StratifiedKFold(n_splits = self.nodes)
        for splited_index in skf.split(X, y):
            # TODO: Scale???? How scale test????
            new_X = pd.DataFrame(X[splited_index[1]])
            new_y = pd.DataFrame(y[splited_index[1]])

            X_file = data_dir + "/data_" + str(node) + ".csv"
            y_file = data_dir + "/class_" + str(node) + ".csv"
            new_X.to_csv(X_file, index = False)
            new_y.to_csv(y_file, index = False)
            node += 1

    @staticmethod
    def connected_geo_graph(nodes):
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

    @staticmethod
    def show_geo_graph(graph, image_dir):
        node_position = nx.get_node_attributes(graph,'pos')

        # Find node near center (0.5,0.5)
        d_min            = 1
        node_near_center = 0
        for node in node_position:
            x, y = node_position[node]
            distance = (x - 0.5)**2 + (y - 0.5)**2
            if distance < d_min:
                node_near_center = node
                d_min = distance

        # Color by path length from node near center
        color_node        = dict(nx.single_source_shortest_path_length(graph, node_near_center))
        array_color_node  = np.array(list(color_node.values()))

        sns.set_style('darkgrid')
        plt.figure(figsize=(8, 8))
        nx.draw_networkx_edges(graph, node_position, nodelist=[node_near_center],alpha=0.4)
        nx.draw_networkx_nodes(graph, node_position, nodelist=color_node.keys(),
                               node_size = 80,
                               node_color = array_color_node,
                               cmap = plt.cm.winter_r)

        plt.xlim(-0.05,1.05)
        plt.ylim(-0.05,1.05)
        plt.axis('off')
        file = image_dir + "/graph.svg"
        plt.savefig(file)

    def __del__(self):
        sub.check_call('rm src/distributedsvm/datas/*.csv', shell = True)
