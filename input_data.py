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

def load_data(dataset):
    path = '/home/huawei/rise/paper_2/dataset/dynamic_datasets/{}.mat'.format(dataset)
    adjs = scio.loadmat(path)['dynamic_dataset']

    features = []
    adjs_ret = []
    for idx in range(adjs.shape[2]):
        adj = adjs[:, :, idx]
        assert len(np.where(np.diag(adj)==1)[0]) == 0
        feature = np.eye(adj.shape[0])  # featureless

        adjs_ret.append(adj)
        features.append(feature)

    return adjs_ret, features
