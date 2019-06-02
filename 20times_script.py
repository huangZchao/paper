import multiprocessing
from metric.compute_map import compute_map
from train import Train_Runner
import tensorflow as tf
import pandas as pd
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
datasets = [['ia-radoslaw-email', 'ia-hospital-ward-proximity', 'ia-primary-school-proximity', 'ia-workplace-contacts'],  # time step == 8
            ['insecta-ant-colony1', 'insecta-ant-colony2', 'insecta-ant-colony4', 'SYN-FIX', 'ca-cit-HepPh'],  # time_step == 10
            ['insecta-ant-colony3', 'insecta-ant-colony5'],  # time_step == 20
            ['insecta-ant-colony6'],  # time_step == 30
            ['ia-contacts_hypertext2009', 'fb-forum', 'SBM'],  # time_step == 7
            ['ca-cit-HepTh'],  # time_step == 15
            ['football']]  # time_step == 5
time_step = [8, 10, 20, 30, 7, 15, 5]



def script(dataname, res, seq_len):
    import settings
    mAPs = []
    for i in range(20):
        tf.reset_default_graph()
        setting = settings.get_settings(dataname, seq_len)
        runner = Train_Runner(setting)
        runner.erun()
        mAP = compute_map(dataname, 'gcn&tcn')
        mAPs.append(mAP)
    average_map = sum(mAPs)/len(mAPs)
    res[dataname] = [average_map]


if __name__ == '__main__':
    save_dict = {}
    procs = []
    with multiprocessing.Manager() as MG:
        res = multiprocessing.Manager().dict()

        for datanames, seq_len in zip(datasets, time_step):
            for dataname in datanames:
                p = multiprocessing.Process(target=script, args=(dataname, res, seq_len-2, ))
                procs.append(p)
                p.start()
        for p in procs:
            p.join()
            print(res)
        for k in res.keys():
            save_dict[k] = res[k]
        print(save_dict)
        res_df = pd.DataFrame(save_dict)
        res_df.to_csv('/home/huawei/PycharmProjects/paper_dataset/result_map/res.csv')  # todo


