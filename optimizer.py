import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(self, struct_preds, temporal_preds, struct_labels, temporal_labels, pos_weights, norms, seq_len):
        self.cost = 0
        for ts in range(seq_len):
            # struct loss
            self.cost += norms[ts] * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=struct_preds[ts],
                                                                                             targets=struct_labels[ts],
                                                                                             pos_weight=pos_weights[ts]))
            self.cost += norms[ts] * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=temporal_preds[ts],
                                                                                             targets=temporal_labels[ts],
                                                                                             pos_weight=pos_weights[ts]))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
