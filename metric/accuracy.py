import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dataname = 'europe-airports'

nodes = np.loadtxt('/home/huawei/risehuang/paper_2/dataset/embedding/{}.emb'.format(dataname))
print nodes.shape
labels = np.loadtxt('/home/huawei/risehuang/paper_2/dataset/struct_similarity/{}/labels-{}.txt'.format(dataname, dataname))

random_state = 2014
# label 0
labels_0 = labels[np.where(labels[:, 1]==0)]
idx_0 = list(map(int, labels_0[:, 0]))
nodes_0 = nodes[idx_0]
labels_0 = labels_0[:, 1]

print len(labels_0)
x_train0, x_test0, y_train0, y_test0 = train_test_split(nodes_0, labels_0, test_size=0.2, random_state=random_state)

# label 1
labels_1 = labels[np.where(labels[:, 1]==1)]
idx_1 = list(map(int, labels_1[:, 0]))
nodes_1 = nodes[idx_1]
labels_1 = labels_1[:, 1]

print len(labels_1)
x_train1, x_test1, y_train1, y_test1 = train_test_split(nodes_1, labels_1, test_size=0.2, random_state=random_state)

# label 2
labels_2 = labels[np.where(labels[:, 1]==2)]
idx_2 = list(map(int, labels_2[:, 0]))
nodes_2 = nodes[idx_2]
labels_2 = labels_2[:, 1]

print len(labels_2)
x_train2, x_test2, y_train2, y_test2 = train_test_split(nodes_2, labels_2, test_size=0.2, random_state=random_state)

# label 3
labels_3 = labels[np.where(labels[:, 1]==3)]
idx_3 = list(map(int, labels_3[:, 0]))
nodes_3 = nodes[idx_3]
labels_3 = labels_3[:, 1]

print len(labels_3)
x_train3, x_test3, y_train3, y_test3 = train_test_split(nodes_3, labels_3, test_size=0.2, random_state=random_state)

x_train = np.concatenate([x_train0, x_train1, x_train2, x_train3], axis=0)
y_train = np.concatenate([y_train0, y_train1, y_train2, y_train3], axis=0)

x_test = np.concatenate([x_test0, x_test1, x_test2, x_test3], axis=0)
y_test = np.concatenate([y_test0, y_test1, y_test2, y_test3], axis=0)


def train(label_val, y_train, y_test):
    y_train = [int(y) if y==label_val else 4 for y in y_train]
    y_test = [int(y) if y==label_val else 4 for y in y_test]

    clf = LogisticRegression(random_state=random_state).fit(x_train, y_train)
    l = clf.predict(x_test)
    proba = clf.predict_proba(x_test)

    return proba[:, 0]

# train
res = []
for l in [0, 1., 2., 3.]:
    res.append(train(l, y_train, y_test)[:, np.newaxis])

res = np.concatenate(res, axis=1)
res = np.argmax(res, axis=1)

acc = 0
for x, y in zip(res, y_test):
    if int(x) == int(y):
        acc += 1

print map(int, res)
print map(int, y_test)
print len(y_test)
print acc/float(len(y_test))
