import numpy as np
import scipy.io as scio
from tqdm import tqdm


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

for data_name in ['fb-forum', 'ia-contacts_hypertext2009', 'ia-radoslaw-email', 'ia-primary-school-proximity','ia-workplace-contacts', 'ia-hospital-ward-proximity']:
    pred = np.loadtxt('/home/huawei/PycharmProjects/paper_dataset/embedding/{}.txt'.format(data_name)) # tcn gcn
    # pred = np.loadtxt('/home/huawei/PycharmProjects/DynamicGEM/emb/rnn/{}.txt'.format(data_name))

    # pred = scio.loadmat('/home/huawei/rise/paper_2/dataset/embedding/{}.mat'.format(data_name))['ans']
    label = scio.loadmat('/home/huawei/PycharmProjects/paper_dataset/dynamic_datasets/{}.mat'.format(data_name))[
                'dynamic_dataset'][:, :, -1]
    print(data_name)

    top_k = int(len(np.where(label==1)[0])/2)
    print('top_k: ', top_k)
    top_k = 5000

    distances = dict()
    for i in tqdm(range(len(pred)-1)):
        for j in range(i+1, len(pred)):
            distances[(i, j)] = np.sum(np.square(pred[i]-pred[j]))

    distances = np.array(sorted(distances.items(), key=lambda item: item[1]))[: top_k]
    pred_edges = [(pair[0], pair[1]) for pair in distances[:, 0]]

    correct = 0
    error = 0
    for i in tqdm(range(len(pred)-1)):
        for j in range(i+1, len(pred)):
            if label[i, j] == 1:
                if (i, j) in pred_edges or (j, i) in pred_edges:
                    correct += 1
                else:
                    error += 1

    print('correct: ', correct)
    print('error: ', error)
    print(pred.shape)


