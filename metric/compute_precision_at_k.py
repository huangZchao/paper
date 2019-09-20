import numpy as np
import scipy.io as scio
from tqdm import tqdm
import networkx as nx

def compute_precision_at_k(data_name, method, look_back, top_k):
    if method == 'gcn&tcn':
        pred = np.loadtxt('/home/huawei/risehuang/paper2/gcn_tcn/embedding/{}.txt'.format(data_name))  # GCN + TCN
    elif method == 'ae':
        pred = np.loadtxt('/home/huawei/PycharmProjects/DynamicGEM/emb/ae/{}.txt'.format(data_name))  # ae
    elif method == 'aernn':
        pred = np.loadtxt('/home/huawei/PycharmProjects/DynamicGEM/emb/aernn/{}.txt'.format(data_name))  # aernn
    elif method == 'rnn':
        pred = np.loadtxt('/home/huawei/PycharmProjects/DynamicGEM/emb/rnn/{}.txt'.format(data_name))  # rnn
    elif method == 'sdne':
        pred = scio.loadmat('/home/huawei/PycharmProjects/SDNE/result/{}/embedding.mat'.format(data_name))['embedding']  # rnn
    else:
        raise ValueError('wrong method.')

    label = scio.loadmat('/home/huawei/risehuang/paper2/gcn_tcn/dynamic_datasets/{}.mat'.format(data_name))[
                'dynamic_dataset'][:, :, look_back]

    distances = dict()
    for i in tqdm(range(len(pred) - 1)):
        for j in range(i + 1, len(pred)):
            distances[(i, j)] = np.sum(np.square(pred[i] - pred[j]))

    distances = np.array(sorted(distances.items(), key=lambda item: item[1]))[: top_k]
    pred_edges = [(pair[0], pair[1]) for pair in distances[:, 0]]

    true_graph = nx.from_numpy_array(label)

    correct = 0
    for src, dst in pred_edges:
        if true_graph.has_edge(src, dst):
            correct += 1

    return correct/top_k