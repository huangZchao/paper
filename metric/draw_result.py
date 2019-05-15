import matplotlib.pyplot as plt
import numpy as np

dataname = 'karate-mirrored'

nodes = np.loadtxt('/home/huawei/risehuang/paper_2/dataset/embedding/{}.emb'.format(dataname))

fig, ax = plt.subplots()

pairs = np.array([1,37,2,39,3,38,4,59,5,63,6,50,7,55,8,43,9,41,10,58,11,53,12,67,13,62,14,60,15,64,16,66,17,52,18,40,19,61,20,54,21,46,22,49,23,47,24,35,25,44,26,57,27,68,28,45,29,56,30,36,31,65,32,48,33,51,34,42])
pairs = np.reshape(pairs, [-1, 2])
for pair in pairs:
    x1 = pair[0]-1
    x2 = pair[1]-1
    ax.scatter(nodes[[x1, x2], 0], nodes[[x1, x2], 1])
n = np.arange(1, len(nodes)+1)
for i, txt in enumerate(n):
    ax.annotate(txt, (nodes[i][0], nodes[i][1]))

# labels = np.loadtxt('/home/huawei/risehuang/paper_2/dataset/struct_similarity/{}/labels-{}.txt'.format(dataname, dataname))
# for value in [0, 1, 2, 3]:
#     idx = list(map(int, labels[np.where(labels[:, 1]==value)][:, 0]))
#     ax.scatter(nodes[idx, 0], nodes[idx, 1])
#
#     for i in idx:
#         ax.annotate(i, (nodes[i][0], nodes[i][1]))

plt.show()