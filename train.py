import tensorflow as tf
from constructor import get_placeholder, format_data, get_model, get_optimizer, update, predict
import numpy as np
import os

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS


class Train_Runner():
    def __init__(self, settings):
        self.data_name = settings['data_name']
        self.iteration = settings['iterations']
        self.train_length = settings['train_length']

    def erun(self):
        # formatted data
        feas = format_data(self.data_name, self.train_length, FLAGS.time_decay)

        # Define placeholders
        placeholders = get_placeholder(feas['struct_adj_norms'][0])

        # construct model
        ae_model = get_model(placeholders, feas['feature_dim'], feas['struct_features_nonzeros'][0],
                             feas['num_node'], self.train_length)

        # Optimizer
        opt = get_optimizer(ae_model, placeholders, self.train_length)

        # Initialize session
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
        sess.run(tf.global_variables_initializer())

        # Train model
        for epoch in range(self.iteration):
            for i in range(1):   # mean only use the first batch;
                avg_cost, feed_dict, struct_loss, temporal_loss = update(opt, sess, feas, i, placeholders)
                print('dataname ', self.data_name, ' epoch ', epoch, 'batch ', i, 'total ', avg_cost, 'struct ', struct_loss, 'temporal ', temporal_loss)

        embeddings = predict(ae_model, sess, feas, placeholders)
        embeddings = np.reshape(np.array(embeddings)[:, -1, :], [feas['num_node'], FLAGS.hidden3[-1]])
        print(embeddings)

        time_decay = FLAGS.time_decay
        alpha = FLAGS.alpha
        emb = FLAGS.hidden3
        emb = '-'.join(list(map(str, emb)))
        subdir = emb+'-'+str(time_decay)+'-'+str(alpha)

        write_path = '/home/huawei/risehuang/paper2/gcn_tcn/embedding/{}/'.format(subdir)
        if not os.path.exists(write_path):
            os.mkdir(write_path)
        write_path = '/home/huawei/risehuang/paper2/gcn_tcn/embedding/{}/{}/'.format(subdir, self.train_length)
        if not os.path.exists(write_path):
            os.mkdir(write_path)

        np.savetxt(write_path + '{}.txt'.format(self.data_name), embeddings)

