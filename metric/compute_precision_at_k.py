import numpy as np
import scipy.io as scio
from tqdm import tqdm
import networkx as nx

def compute_precision_at_k(data_name, method, look_back, top_k):
    dataset = scio.loadmat('/home/huawei/risehuang/paper2/gcn_tcn/dynamic_datasets/{}.mat'.format(data_name))[
                'dynamic_dataset']
    label = dataset[:, :, look_back]
    pred = np.loadtxt('/home/huawei/risehuang/paper2/gcn_tcn/embedding/{}/{}.txt'.format(dataset.shape[2]+look_back-1, data_name))  # GCN + TCN

    distances = dict()
    for i in tqdm(range(len(pred) - 1)):
        for j in range(i + 1, len(pred)):
            distances[(i, j)] = np.sum(np.square(pred[i] - pred[j]))

    distances = np.array(sorted(distances.items(), key=lambda item: item[1]))[: top_k]
    pred_edges = [(pair[0], pair[1]) for pair in distances[:, 0]]
    print(distances)
    true_graph = nx.from_numpy_array(label)

    correct = 0
    for src, dst in pred_edges:
        if true_graph.has_edge(src, dst):
            correct += 1

    return correct/top_k