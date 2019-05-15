import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

dataname = 'karate-mirrored'

# orig_graph = []
# path = '/home/huawei/risehuang/paper_2/dataset/struct_similarity/{}/{}.edgelist'.format(dataname, dataname)
# with open(path) as file:
#     for line in file:
#         head, tail = [int(x) for x in line.split()]
#         orig_graph.append([head, tail])
#
# orig_graph = np.array(orig_graph)
# n = np.arange(np.max(orig_graph))
# print(n)
# fig, ax = plt.subplots()
# ax.scatter(orig_graph[:, 0], orig_graph[:, 1])
#
# for i, txt in enumerate(n):
#     ax.annotate(txt, (orig_graph[i][0], orig_graph[i][1]))

nodes = np.loadtxt('/home/huawei/risehuang/paper_2/dataset/embedding/{}.emb'.format(dataname))

fig, ax = plt.subplots()

pairs = np.array([1,37,2,39,3,38,4,59,5,63,6,50,7,55,8,43,9,41,10,58,11,53,12,67,13,62,14,60,15,64,16,66,17,52,18,40,19,61,20,54,21,46,22,49,23,47,24,35,25,44,26,57,27,68,28,45,29,56,30,36,31,65,32,48,33,51,34,42])
pairs = np.reshape(pairs, [-1, 2])
for pair in pairs:
    x1 = pair[0]-1
    x2 = pair[1]-1
    print(x1, x2)
    ax.scatter(nodes[[x1, x2], 0], nodes[[x1, x2], 1])

n = np.arange(1, len(nodes)+1)
for i, txt in enumerate(n):
    ax.annotate(txt, (nodes[i][0], nodes[i][1]))

plt.show()