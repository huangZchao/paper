import tensorflow as tf
import numpy as np
from model import GCN
from optimizer import OptimizerAE
from input_data import load_data
from preprocessing import preprocess_graph, sparse_to_tuple, construct_feed_dict
flags = tf.app.flags
FLAGS = flags.FLAGS

def get_placeholder(adj_norms):
    placeholders = {
        'features': [tf.sparse_placeholder(tf.float32, name='features_{}'.format(i)) for i in range(len(adj_norms))],
        'adj_norms': [tf.sparse_placeholder(tf.float32, name='adj_norms_{}'.format(i)) for i in range(len(adj_norms))],
        'adj_origs': [tf.sparse_placeholder(tf.float32, name='adj_origs_{}'.format(i)) for i in range(len(adj_norms))],
        'dropout': tf.placeholder_with_default(0., shape=()),  # todo
    }

    return placeholders


def get_model(placeholders, feature_dim, features_nonzeros, num_node, seq_len, num_channel):
    model = GCN(placeholders, feature_dim, features_nonzeros, num_node, seq_len, num_channel)
    return model

def format_data(data_name):
    # Load data
    adjs, features = load_data(data_name)

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

    feas = {'adj_origs': adj_origs,
            'features': features_sp,
            'adj_norms': adj_norms,
            'num_node': num_node,
            'feature_dim': feature_dim,
            'features_nonzeros': features_nonzeros,
            'pos_weights': pos_weights,
            'norms': norms}

    return feas

def get_optimizer(model, placeholders, pos_weights, norms, seq_len):
    opt = OptimizerAE(struct_preds=model.reconstructions,
                      temporal_preds=model.reconstructions_tss,
                      struct_labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_origs'],
                                                                  validate_indices=False), [-1]),
                      temporal_labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_origs'],
                                                                  validate_indices=False), [-1]),
                      pos_weights=pos_weights,
                      norms=norms,
                      seq_len=seq_len)
    return opt

def update(model, opt, sess, adj_norms, adj_label, features, placeholders):
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norms, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    emb = sess.run(model.embeddings, feed_dict=feed_dict)

    _, reconstruct_loss = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)

    return emb, reconstruct_loss, feed_dict

