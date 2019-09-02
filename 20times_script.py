import multiprocessing
from metric.compute_map import compute_map
from metric.compute_precision_at_k import *
from train import Train_Runner
import tensorflow as tf
import pandas as pd
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# datasets = [
#     ['fb-forum', 'ia-contacts_hypertext2009', 'ia-radoslaw-email', 'ia-primary-school-proximity'],
#     ['ia-workplace-contacts', 'ia-hospital-ward-proximity']]
# time_step = [7, 8]
datasets = [
    ['ia-primary-school-proximity']]
time_step = [7]


def script(dataname, res, seq_len, look_back):
    import settings
    mAPs = []
    tops = {100: [], 500: [], 1000: []}
    for i in range(1):
        tf.reset_default_graph()
        setting = settings.get_settings(dataname, seq_len - look_back)
        runner = Train_Runner(setting)
        start = time.clock()
        runner.erun()
        print(dataname, " time consuming: ", time.clock() - start)

        mAP = compute_map(dataname, 'gcn&tcn', -(look_back-1))
        mAPs.append(mAP)
    average_map = sum(mAPs)/len(mAPs)
    res[dataname] = [average_map]

    #     for top in [100, 500, 1000]:
    #         tops[top].append(compute_precision_at_k(dataname, 'gcn&tcn', -(look_back-1), top))
    # for top, v in tops.items():
    #     average_precision = sum(v)/len(v)
    #     res[dataname+'_top_{}'.format(top)] = [average_precision]


if __name__ == '__main__':
    save_dict = {}
    procs = []
    for look_back in range(2, 3):
        with multiprocessing.Manager() as MG:
            res = multiprocessing.Manager().dict()

            for datanames, seq_len in zip(datasets, time_step):
                for dataname in datanames:
                    p = multiprocessing.Process(target=script, args=(dataname, res, seq_len, look_back, ))
                    procs.append(p)
                    p.start()
            for p in procs:
                p.join()
                print(res)
            for k in res.keys():
                save_dict[k] = res[k]
            print(save_dict)
            res_df = pd.DataFrame(save_dict)
            res_df.to_csv('/home/huawei/PycharmProjects/paper_dataset/result_map/res_{}.csv'.format(look_back))  # todo


