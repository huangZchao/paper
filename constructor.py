import tensorflow as tf
import numpy as np
from model import GCN
from optimizer import OptimizerAE
from input_data import load_data
from preprocessing import preprocess_graph, sparse_to_tuple, construct_feed_dict
flags = tf.app.flags
FLAGS = flags.FLAGS

def get_placeholder(struct_adj_norms):
    placeholders = {
        'struct_features': [tf.sparse_placeholder(tf.float32, name='struct_features_{}'.format(i)) for i in
                            range(len(struct_adj_norms))],
        'struct_adj_norms': [tf.sparse_placeholder(tf.float32, name='struct_adj_norms_{}'.format(i)) for i in
                             range(len(struct_adj_norms))],
        'struct_adj_origs': [tf.sparse_placeholder(tf.float32, name='struct_adj_origs{}'.format(i)) for i in
                             range(len(struct_adj_norms))],

        'temporal_adj_origs': [tf.sparse_placeholder(tf.float32, name='temporal_adj_origs_{}'.format(i)) for i in
                               range(len(struct_adj_norms))],

        'struct_pos_weights': tf.placeholder(dtype=tf.float32, shape=[len(struct_adj_norms), ], name='struct_pos_weights'),
        'struct_norms': tf.placeholder(dtype=tf.float32, shape=[len(struct_adj_norms), ], name='struct_norms'),
        'temporal_pos_weights': tf.placeholder(dtype=tf.float32, shape=[len(struct_adj_norms), ], name='temporal_pos_weights'),
        'temporal_norms': tf.placeholder(dtype=tf.float32, shape=[len(struct_adj_norms), ], name='temporal_norms'),

        'dropout': tf.placeholder_with_default(0., shape=()),  # todo
    }

    return placeholders


def get_model(placeholders, feature_dim, features_nonzeros, num_node, seq_len):
    model = GCN(placeholders, feature_dim, features_nonzeros, num_node, seq_len)
    return model

def format_data(data_name, seq_len, time_decay):
    # Load data
    adjs, features = load_data(data_name, time_decay)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_origs = []
    pos_weights = []
    norms = []
    adj_norms = []
    features_sp = []
    features_nonzeros = []

    num_node = np.array(adjs[0]).shape[1]
    feature_dim = np.array(features[0]).shape[1]

    for adj, feature in zip(adjs, features):
        adj_orig = sparse_to_tuple(adj)

        pos_weight = float(num_node * num_node - adj_orig[1].sum()) / adj_orig[1].sum()
        norm = num_node * num_node / float((num_node * num_node - adj_orig[1].sum()) * 2)

        feature = sparse_to_tuple(feature)
        features_nonzero = feature[1].shape[0]

        adj_norm = preprocess_graph(adj)

        adj_origs.append(adj_orig)
        pos_weights.append(pos_weight)
        norms.append(norm)
        features_sp.append(feature)
        features_nonzeros.append(features_nonzero)
        adj_norms.append(adj_norm)

    batch_size = len(adj_origs) - seq_len

    temporal_adj_origs = []
    temporal_pos_weights = []
    temporal_norms = []

    struct_adj_origs = []
    struct_pos_weights = []
    struct_norms = []
    struct_adj_norms = []
    struct_features = []
    struct_features_nonzeros = []

    for i in range(batch_size):
        temporal_adj_origs.append(adj_origs[i+1: i+1+seq_len])
        temporal_pos_weights.append(pos_weights[i+1: i+1+seq_len])
        temporal_norms.append(norms[i+1: i+1+seq_len])

        struct_adj_origs.append(adj_origs[i: i+seq_len])
        struct_pos_weights.append(pos_weights[i: i+seq_len])
        struct_norms.append(norms[i: i+seq_len])
        struct_adj_norms.append(adj_norms[i: i+seq_len])
        struct_features.append(features_sp[i: i+seq_len])
        struct_features_nonzeros.append(features_nonzeros[i: i+seq_len])

    # temporal_adj_origs = adj_origs[1: 1+seq_len]
    # temporal_pos_weights = pos_weights[1: 1+seq_len]
    # temporal_norms = norms[1: 1+seq_len]
    #
    # struct_adj_origs = adj_origs[0: 0+seq_len]
    # struct_pos_weights = pos_weights[0: 0+seq_len]
    # struct_norms = norms[0: 0+seq_len]
    # struct_adj_norms = adj_norms[0: 0+seq_len]
    # struct_features = features_sp[0: 0+seq_len]
    # struct_features_nonzeros = features_nonzeros[0: 0+seq_len]

    feas = {'temporal_adj_origs': temporal_adj_origs,
            'temporal_pos_weights': temporal_pos_weights,
            'temporal_norms': temporal_norms,

            'num_node': num_node,
            'feature_dim': feature_dim,
            'batch_size': batch_size,

            'struct_adj_origs': struct_adj_origs,
            'struct_features': struct_features,
            'struct_features_nonzeros': struct_features_nonzeros,
            'struct_adj_norms': struct_adj_norms,
            'struct_pos_weights': struct_pos_weights,
            'struct_norms': struct_norms,

            'adj_norms': adj_norms,
            'features': features_sp
            }

    return feas

def get_optimizer(model, placeholders, seq_len):
    opt = OptimizerAE(struct_preds=model.reconstructions,
                      temporal_preds=model.reconstructions_tss,
                      struct_labels=placeholders['struct_adj_origs'],
                      temporal_labels=placeholders['temporal_adj_origs'],
                      struct_pos_weights=placeholders['struct_pos_weights'],
                      struct_norms=placeholders['struct_norms'],
                      temporal_pos_weights=placeholders['temporal_pos_weights'],
                      temporal_norms=placeholders['temporal_norms'],
                      seq_len=seq_len)
    return opt

def update(opt, sess, feas, i, placeholders):
    struct_adj_norms = feas['struct_adj_norms'][i]
    struct_adj_origs = feas['struct_adj_origs'][i]
    struct_features = feas['struct_features'][i]

    temporal_adj_origs = feas['temporal_adj_origs'][i]

    struct_pos_weights = feas['struct_pos_weights'][i]
    struct_norms = feas['struct_norms'][i]
    temporal_pos_weights = feas['temporal_pos_weights'][i]
    temporal_norms = feas['temporal_norms'][i]

    # Construct feed dictionary
    feed_dict = construct_feed_dict(struct_adj_norms, struct_adj_origs, struct_features, temporal_adj_origs,
                                    struct_pos_weights, struct_norms, temporal_pos_weights, temporal_norms,
                                    placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    _, reconstruct_loss, struct_cost, temporal_cost = sess.run([opt.opt_op, opt.cost, opt.struct_cost, opt.temporal_cost],
                                                               feed_dict=feed_dict)

    return reconstruct_loss, feed_dict, struct_cost, temporal_cost

def predict(model, sess, feas, placeholders):
    batch_size = feas['batch_size']
    adj_norms = feas['struct_adj_norms'][batch_size-1]
    features = feas['struct_features'][batch_size-1]
    # adj_norms = feas['adj_norms'][2: 7]
    # features = feas['features'][2: 7]

    feed_dict = dict()
    for i, d in zip(placeholders['struct_adj_norms'], adj_norms):
        feed_dict.update({i: d})
    for i, d in zip(placeholders['struct_features'], features):
        feed_dict.update({i: d})
    return sess.run(model.sequence_out, feed_dict=feed_dict)

# if __name__ == '__main__':
#     format_data('SYN-VAR', 3)