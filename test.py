import scipy.io as scio

network = scio.loadmat('/home/huawei/risehuang/paper_2/dataset/dynamic datasets/football.mat')
tmp1 = network['dynamic_dataset']

print(len(tmp1))
print(len(tmp1[0]))
print(len(tmp1[0][0]))

print(tmp1[:, :, 0])