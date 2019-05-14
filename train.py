import os
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
from constructor import get_placeholder, get_model, format_data, get_optimizer, update

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS


class Train_Runner():
    def __init__(self, settings):
        self.data_name = settings['data_name']
        self.iteration = settings['iterations']

    def erun(self):
        # formatted data
        feas = format_data(self.data_name)

        # Define placeholders
        placeholders = get_placeholder(feas['adj'])

        # construct model
        d_real, discriminator, ae_model = get_model(placeholders, feas['num_features'], feas['features_nonzero'])

        # Optimizer
        opt = get_optimizer(ae_model, discriminator, placeholders, feas['pos_weight'], feas['norm'], d_real)

        # Initialize session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Train model
        for epoch in range(self.iteration):
            emb, avg_cost = update(ae_model, opt, sess, feas['adj_norm'], feas['adj'], feas['features'], placeholders, feas['adj'])
            print(avg_cost)
        print(emb)

