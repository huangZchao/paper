import numpy as np
import scipy.io as scio
from tqdm import tqdm
from sklearn import cross_validation,metrics


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

for data_name in ['fb-forum', 'ia-contacts_hypertext2009', 'ia-radoslaw-email', 'ia-primary-school-proximity','ia-workplace-contacts', 'ia-hospital-ward-proximity']:
    # pred = np.loadtxt('/home/huawei/PycharmProjects/paper_dataset/embedding/{}.txt'.format(data_name)) # tcn gcn
    # pred = np.loadtxt('/home/huawei/PycharmProjects/DynamicGEM/emb/rnn/{}.txt'.format(data_name)) # rnn
    # pred = np.loadtxt('/home/huawei/PycharmProjects/DynamicGEM/emb/ae/{}.txt'.format(data_name))  # ae
    pred = np.loadtxt('/home/huawei/PycharmProjects/DynamicGEM/emb/aernn/{}.txt'.format(data_name))  # aernn


    label = scio.loadmat('/home/huawei/PycharmProjects/paper_dataset/dynamic_datasets/{}.mat'.format(data_name))[
                'dynamic_dataset'][:, :, -1]
    print(data_name)

    score_pos = []
    score_neg = []
    for i in tqdm(range(len(pred)-1)):
        for j in range(i+1, len(pred)):
            distance = np.sum(np.square(pred[i]-pred[j]))
            if label[i, j] == 1:
                score_pos.append(distance)
            else:
                score_neg.append(distance)

    max_dist = max(max(score_pos), max(score_neg))
    score = score_pos + score_neg
    score = [s/max_dist for s in score]
    labels = [1]*len(score_pos) + [0]*len(score_neg)

    test_auc = metrics.roc_auc_score(labels, score)

    print("auc: ", test_auc)


