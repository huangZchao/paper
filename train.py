import tensorflow as tf
# from constructor import get_placeholder, get_model, format_data, get_optimizer, update
from constructor import get_placeholder, format_data, get_model

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
        feas = format_data(self.data_name)
        # feas = {'adj_origs': adj_origs,
        #         'features': features_sp,
        #         'adj_norms': adj_norms,
        #         'num_node': num_node,
        #         'feature_dim': feature_dim,
        #         'features_nonzeros': features_nonzeros,
        #         'pos_weights': pos_weights,
        #         'norms': norms}

        # Define placeholders
        placeholders = get_placeholder(feas['adj_norms'])

        # construct model
        ae_model = get_model(placeholders, feas['feature_dim'], feas['features_nonzeros'],
                             feas['num_node'], self.seq_len, FLAGS.num_channel)

        # # Optimizer
        # opt = get_optimizer(ae_model, placeholders, feas['pos_weights'], feas['norms'])
        #
        # # Initialize session
        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        #
        # # Train model
        # for epoch in range(self.iteration):
        #     emb, avg_cost, feed_dict = update(ae_model, opt, sess, feas['adj_norms'], feas['adj'], feas['features'], placeholders, feas['adj'])
        #     print avg_cost
        #
        # # print sess.run([ae_model.alphas], feed_dict=feed_dict)
        # np.savetxt('/home/huawei/risehuang/paper_2/dataset/embedding/{}.emb'.format(self.data_name), emb)

