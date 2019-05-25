import tensorflow as tf
from constructor import get_placeholder, format_data, get_model, get_optimizer, update, predict
import numpy as np
from tensorflow.core.protobuf import config_pb2

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS


class Train_Runner():
    def __init__(self, settings):
        self.data_name = settings['data_name']
        self.iteration = settings['iterations']
        self.seq_len = settings['seq_len']

    def erun(self):
        # formatted data
        feas = format_data(self.data_name, self.seq_len)

        # Define placeholders
        placeholders = get_placeholder(feas['struct_adj_norms'][0])

        # construct model
        ae_model = get_model(placeholders, feas['feature_dim'], feas['struct_features_nonzeros'][0],
                             feas['num_node'], self.seq_len)

        # Optimizer
        opt = get_optimizer(ae_model, placeholders, self.seq_len)

        # Initialize session
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())

        # Train model
        for epoch in range(self.iteration):
            for i in range(feas['batch_size']-1):
                avg_cost, feed_dict, struct_loss, temporal_loss = update(opt, sess, feas, i, placeholders)
                print('epoch ', epoch, 'batch ', i, 'total ', avg_cost, 'struct ', struct_loss, 'temporal ', temporal_loss)

        embeddings = predict(ae_model, sess, feas, placeholders)
        embeddings = np.reshape(np.array(embeddings)[:, -1, :], [feas['num_node'], FLAGS.hidden3[-1]])
        print(embeddings)
        np.savetxt('/home/huawei/rise/paper_2/dataset/embedding/{}.txt'.format(self.data_name), embeddings)

