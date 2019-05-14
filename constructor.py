import tensorflow as tf
import numpy as np
from model import ARGA, Discriminator
from optimizer import OptimizerAE
import scipy.sparse as sp
from input_data import load_data
from preprocessing import preprocess_graph, sparse_to_tuple, mask_test_edges, construct_feed_dict
flags = tf.app.flags
FLAGS = flags.FLAGS

def get_placeholder(adj):
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'real_distribution': tf.placeholder(dtype=tf.float32, shape=[adj[2][1], FLAGS.hidden2],
                                            name='real_distribution')
    }

    return placeholders


def get_model(placeholders, num_features, features_nonzero):
    discriminator = Discriminator()
    d_real = discriminator.construct(placeholders['real_distribution'])
    model = ARGA(placeholders, num_features, features_nonzero)

    return d_real, discriminator, model

# todo try one layer
def format_data(data_name):
    # Load data
    adj, features, adj_orig = load_data(data_name, layer=0)

    # Store original adjacency matrix (without diagonal entries) for later
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj.shape)

    if FLAGS.features == 0:
        features = sp.identity(features.shape[0])  # featureless # todo why is identity

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj = sparse_to_tuple(adj)
    adj_orig = sparse_to_tuple(adj_orig)
    num_nodes = adj[2][1]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    pos_weight = float(num_nodes * num_nodes - adj[1].sum()) / adj[1].sum()
    norm = num_nodes * num_nodes / float((num_nodes * num_nodes - adj[1].sum()) * 2)

    feas = {'adj': adj_orig,
            'features': features,
            'adj_norm': adj_norm,
            'num_nodes': num_nodes,
            'num_features': num_features,
            'features_nonzero': features_nonzero,
            'pos_weight': pos_weight,
            'norm': norm}

    return feas

def get_optimizer(model, discriminator, placeholders, pos_weight, norm, d_real):
    d_fake = discriminator.construct(model.embeddings, reuse=True)
    opt = OptimizerAE(preds=model.reconstructions,
                      labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                  validate_indices=False), [-1]),
                      pos_weight=pos_weight,
                      norm=norm,
                      d_real=d_real,
                      d_fake=d_fake)
    return opt

def update(model, opt, sess, adj_norm, adj_label, features, placeholders, adj):
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    feed_dict.update({placeholders['dropout']: 0})
    emb = sess.run(model.embeddings, feed_dict=feed_dict)

    z_real_dist = np.random.randn(adj[2][1], FLAGS.hidden2)
    feed_dict.update({placeholders['real_distribution']: z_real_dist})

    # train 5 epoch for every one epoch in d&g
    for j in range(5):
        _, reconstruct_loss = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
    d_loss, _ = sess.run([opt.dc_loss, opt.discriminator_optimizer], feed_dict=feed_dict)
    g_loss, _ = sess.run([opt.generator_loss, opt.generator_optimizer], feed_dict=feed_dict)

    avg_cost = reconstruct_loss

    return emb, avg_cost


# def retrieve_name(var):
#     callers_local_vars = inspect.currentframe().f_back.f_locals.items()
#     print([var_name for var_name, var_val in callers_local_vars if var_val is var])
#     return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]
