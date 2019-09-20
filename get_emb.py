import multiprocessing
from metric.compute_map import compute_map
from metric.compute_precision_at_k import *
from train import Train_Runner
import tensorflow as tf
import pandas as pd
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

datasets = [
    ['fb-forum', 'ia-contacts_hypertext2009', 'ia-radoslaw-email', 'ia-primary-school-proximity'],
    ['ia-workplace-contacts', 'ia-hospital-ward-proximity']]
time_step = [7, 8]
# datasets = [
#     ['ia-primary-school-proximity']]
# time_step = [7]


def script(dataname, seq_len, look_back):
    import settings

    for i in range(1):
        tf.reset_default_graph()
        setting = settings.get_settings(dataname, seq_len - look_back)  #
        runner = Train_Runner(setting)
        start = time.clock()
        runner.erun()
        print(dataname, " time consuming: ", time.clock() - start)



if __name__ == '__main__':
    save_dict = {}
    procs = []
    for look_back in range(2, 3):
        with multiprocessing.Manager() as MG:

            for datanames, seq_len in zip(datasets, time_step):
                for dataname in datanames:
                    p = multiprocessing.Process(target=script, args=(dataname, seq_len, look_back, ))
                    procs.append(p)
                    p.start()
            for p in procs:
                p.join()




