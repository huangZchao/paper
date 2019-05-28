import numpy as np
import scipy.io as scio


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

def load_data(dataset, time_decay):
    path = '/home/huawei/PycharmProjects/paper_dataset/dynamic_datasets/{}.mat'.format(dataset)
    adjs = scio.loadmat(path)['dynamic_dataset']

    features = []
    adjs_ret = []
    for idx in range(adjs.shape[2]):
        adj = adjs[:, :, idx]
        for i in range(len(adj)):
            adj[i, i] = 0
        feature = np.eye(adj.shape[0]) + adj  # featureless
        adj = np.eye(adj.shape[0]) + adj
        if idx:
            feature += (time_decay * features[idx-1]) * feature

        adjs_ret.append(adj)
        features.append(feature)

    return adjs_ret, features

if __name__ == '__main__':
    load_data('SYN-VAR')
