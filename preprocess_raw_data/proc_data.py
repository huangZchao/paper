import networkx as nx
import numpy as np
import scipy.io as scio


def reorder_data(data_name, threshold=2000):
    ordered = dict()
    nodes = set()
    edges = []

    with open('/home/huawei/risehuang/paper2/gcn_tcn/dynamic_datasets/raw_data/' + data_name + '.edges') as f:
        for line in f.readlines():
            try:
                tmp = line.split(' ')
                src = int(tmp[0])
                dst = int(tmp[1])
            except:
                tmp = line.split(',')
                src = int(tmp[0])
                dst = int(tmp[1])

            nodes.add(src)
            nodes.add(dst)
            edges.append([src, dst])
    nodes = list(nodes)
    for idx, node in enumerate(nodes):
        ordered[node] = idx + 1
    print('num_nodes: ', max(ordered.values()))

    with open('/home/huawei/risehuang/paper2/gcn_tcn/dynamic_datasets/raw_data/' + data_name + '_new.edges', 'w') as f:
        for edge in edges:
            src = ordered[edge[0]]
            dst = ordered[edge[1]]
            if src > threshold or dst > threshold:
                continue
            f.write(str(src) + ' ' + str(dst) + '\n')
    return max(ordered.values())


def split_data(data_name, timesteps, num_nodes):
    main_dir = '/home/huawei/risehuang/paper2/gcn_tcn/dynamic_datasets/raw_data/'
    filename = main_dir + data_name + '_new.edges'

    with open(filename, 'r') as f:
        lines = f.readlines()
        L = len(lines)
        print('total edges cnt: ', L)
        each_step_len = int(L / timesteps)
        graphs = np.zeros([num_nodes, num_nodes, timesteps])

        for i in range(timesteps):
            G = np.zeros((num_nodes, num_nodes))
            shift = (i + 1) * each_step_len
            # todo 不叠加
            # for j in range(shift-each_step_len, shift):
            for j in range(shift):
                line = lines[j]
                line = line.split(' ')
                if int(line[0]) > num_nodes or int(line[1]) > num_nodes or int(line[0]) == int(line[1]):
                    continue
                else:
                    G[int(line[0]) - 1][int(line[1]) - 1] = 1
                    G[int(line[1]) - 1][int(line[0]) - 1] = 1
            print(len(np.where(G != 0)[0]) / 2)
            graphs[:, :, i] = G.astype(np.int)

        print("time length: ", graphs.shape[2])
        output = "/home/huawei/risehuang/paper2/gcn_tcn/dynamic_datasets/" + data_name + '.mat'
        scio.savemat(output, {'dynamic_dataset': graphs})


def analysis(dataset):
    path = '/home/huawei/risehuang/paper2/gcn_tcn/dynamic_datasets/{}.mat'.format(dataset)
    adjs = scio.loadmat(path)['dynamic_dataset']

    adjs_ret = []
    diff = []
    for idx in range(adjs.shape[2]):
        adj = adjs[:, :, idx]
        for i in range(len(adj)):
            adj[i, i] = 0

        adjs_ret.append(adj)
        print('step ', idx, ' edges cnt ', len(np.where(adj != 0)[0]) / 2)

    for i in range(1, len(adjs_ret) - 1):
        diff.append(adjs_ret[i] - adjs_ret[i - 1])

    for d in diff:
        print('change edges cnt ', len(np.where(d != 0)[0]) / 2)


if __name__ == '__main__':
    data_name = 'ia-yahoo-messages'
    time_steps = 7
    threshold = 3000
    nums_nodes = reorder_data(data_name, threshold)
    nums_nodes = min(nums_nodes, threshold)
    print('final num nodes: ', nums_nodes)

    split_data(data_name, time_steps, nums_nodes)
    analysis(data_name)
