import scipy.io as scio
import numpy as np

dataset = 'SYN-FIX'
path = '/home/huawei/PycharmProjects/paper_dataset/dynamic_datasets/{}.mat'.format(dataset)
adjs = scio.loadmat(path)['dynamic_dataset']

adjs_ret = []
diff = []
for idx in range(adjs.shape[2]):
    adj = adjs[:, :, idx]
    if len(np.where(np.diag(adj) == 1)[0]) != 0:
        for i in range(len(adj)):
            adj[i, i] = 0

    adjs_ret.append(adj)

for i in range(1, len(adjs_ret)-1):
    diff.append(adjs_ret[i]-adjs_ret[i-1])

for d in diff:
    print(d)