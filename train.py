import tensorflow as tf
from constructor import get_placeholder, format_data, get_model, get_optimizer, update, predict
import numpy as np

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
        placeholders = get_placeholder(feas['struct_adj_norms'])

        # construct model
        ae_model = get_model(placeholders, feas['feature_dim'], feas['struct_features_nonzeros'],
                             feas['num_node'], self.seq_len, FLAGS.num_channel)

        # Optimizer
        opt = get_optimizer(ae_model, placeholders, feas, self.seq_len)

        # Initialize session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Train model
        for epoch in range(self.iteration):
            avg_cost, feed_dict, struct_loss, temporal_loss = update(opt, sess, feas['struct_adj_norms'], feas['struct_adj_origs'],
                                              feas['struct_features'], feas['temporal_adj_origs'], placeholders)
            print('total ', avg_cost, 'struct ', struct_loss, 'temporal ', temporal_loss)

        embeddings = predict(ae_model, sess, feas, self.seq_len, placeholders)
        print(embeddings)
        embeddings = np.reshape(np.array(embeddings)[0, -1, :], [feas['num_node'], FLAGS.hidden2])
        np.savetxt('/home/huawei/rise/paper_2/dataset/embedding/{}.txt'.format(self.data_name), embeddings)

