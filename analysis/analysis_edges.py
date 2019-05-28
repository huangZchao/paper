import scipy.io as scio
import numpy as np

dataset = 'SBM'
path = '/home/huawei/PycharmProjects/paper_dataset/dynamic_datasets/{}.mat'.format(dataset)
adjs = scio.loadmat(path)['dynamic_dataset']

adjs_ret = []
diff = []
for idx in range(adjs.shape[2]):
    adj = adjs[:, :, idx]
    for i in range(len(adj)):
        adj[i, i] = 0

    adjs_ret.append(adj)
    print('step ', idx, ' edges cnt ', len(np.where(adj!=0)[0])/2)

for i in range(1, len(adjs_ret)-1):
    diff.append(adjs_ret[i]-adjs_ret[i-1])

for d in diff:
    print('change edges cnt ', len(np.where(d!=0)[0])/2)