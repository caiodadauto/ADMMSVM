import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import subprocess as sub
import matplotlib.pyplot as plt
from pathconf import datas_path, graph_path
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

class Network(object):
    def __init__(self, nodes):
        if type(nodes) == str:
            # TODO
            print('Under construction!!')
            exit(0)
        else:
            self.nodes = nodes
            self.graph = self.connected_geo_graph(self.nodes)
            self.show_geo_graph(self.graph)

    def create_graph_mpi(self):
        index        = []
        neighborhood = []
        sum_degree   = 0
        path_file    = graph_path.joinpath('graph_mpi.csv')
        node_degrees = self.graph.degree_iter()
        for node, degree in node_degrees:
            sum_degree += degree
            index.append(sum_degree)
            node_neighbors = self.graph.neighbors_iter(node)
            for neighbor in node_neighbors:
                neighborhood.append(neighbor)

        df = pd.DataFrame({'index': pd.Series(index),
                           'neighborhood':pd.Series(neighborhood)})
        df.to_csv(path_file, index=False)

    def split_data(self, X, y, stratified = True, bad_chess = False):
        if bad_chess:
            n_points = int(X.shape[0] / self.nodes)
            for node in range(self.nodes):
                start_slice = node * n_points
                final_slice = start_slice + n_points
                dx = X[start_slice:final_slice]
                dy = y[start_slice:final_slice]

                frame_dx = pd.DataFrame(dx)
                frame_dy = pd.DataFrame(dy)

                file_data  = datas_path.joinpath('data_' + str(node) + '.csv')
                file_class = datas_path.joinpath('class_' + str(node) + '.csv')
                frame_dx.to_csv(file_data, index = False)
                frame_dy.to_csv(file_class, index = False)
        else:
            node = 0
            if stratified:
                skf  = StratifiedKFold(n_splits = self.nodes)
            else:
                skf  = KFold(n_splits = self.nodes, shuffle = True, random_state = 17)
            for splited_index in skf.split(X, y):
                new_X = pd.DataFrame(X[splited_index[1]])
                new_y = pd.DataFrame(y[splited_index[1]])

                X_path = datas_path.joinpath("data_" + str(node) + ".csv")
                y_path = datas_path.joinpath("class_" + str(node) + ".csv")
                new_X.to_csv(X_path, index = False)
                new_y.to_csv(y_path, index = False)
                node += 1

    def clean_files(self):
        if list(datas_path.glob('*.csv')):
            command = "rm " + str(datas_path) + "/*.csv"
            sub.check_call(command, shell = True)

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
    def show_geo_graph(graph):
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
        cmap = sns.cubehelix_palette(start = .5, rot = -.65, dark = .4, light = .6, as_cmap = True)
        plt.figure(figsize = (10, 8))
        nx.draw_networkx_edges(graph, node_position, nodelist=[node_near_center],alpha=0.4)
        nx.draw_networkx_nodes(graph, node_position, nodelist=color_node.keys(),
                               node_size = 80,
                               node_color = array_color_node,
                               cmap = cmap)

        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.axis('off')
        file = str(graph_path) + "/graph.pdf"
        plt.savefig(file, transparent = True)

    def __del__(self):
        self.clean_files()
