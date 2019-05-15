import numpy as np
import pickle as pkl
import networkx as nx


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset, hop_num):

    orig_graph = nx.Graph()
    path = '/home/huawei/risehuang/paper_2/dataset/struct_similarity/{}/{}.edgelist'.format(dataset, dataset)
    with open(path) as file:
        for line in file:
            head, tail = [int(x) for x in line.split()]
            orig_graph.add_edge(head, tail)
    adj_orig = nx.adjacency_matrix(orig_graph)

    features = adj_orig

    adjs = []
    for i in range(hop_num):
        graph = nx.Graph()
        for node in range(1, nx.number_of_nodes(orig_graph)+1):
            graph.add_node(node)
        path = '/home/huawei/risehuang/paper_2/dataset/struct_similarity/{}/weights_distances-layer-{}.pickle'.format(dataset, i)
        graph_dict = pkl.load(open(path, 'rb'))
        for k, v in graph_dict.items():
            graph.add_weighted_edges_from([(int(k[0]), int(k[1]), v)])

        adj = nx.adjacency_matrix(graph)
        adjs.append(adj)

    return adjs, features, adj_orig
