import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(self, struct_preds, temporal_preds, struct_labels, temporal_labels,
                 struct_pos_weights, struct_norms,
                 temporal_pos_weights, temporal_norms,
                 seq_len):
        self.struct_cost = 0
        self.temporal_cost = 0
        for ts in range(seq_len):
            # struct loss
            struct_label = tf.reshape(tf.sparse_tensor_to_dense(struct_labels[ts],
                                                 validate_indices=False), [-1])
            temporal_label = tf.reshape(tf.sparse_tensor_to_dense(temporal_labels[ts],
                                                 validate_indices=False), [-1])

            self.struct_cost += struct_norms[ts] * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=struct_preds[ts],
                                                                                             targets=struct_label,
                                                                                             pos_weight=struct_pos_weights[ts]))
            self.temporal_cost += temporal_norms[ts] * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=temporal_preds[ts],
                                                                                             targets=temporal_label,
                                                                                             pos_weight=temporal_pos_weights[ts]))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.cost = self.struct_cost + self.temporal_cost
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
