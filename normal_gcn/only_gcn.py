import tensorflow as tf
from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, Dense
import numpy as np
from input_data import load_data
from preprocessing import sparse_to_tuple, preprocess_graph


def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise

# flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_name', 'SYN-VAR', 'name of data set.')
flags.DEFINE_float('learning_rate', .5*0.001, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('features', 0, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('seed', 50, 'seed for fixing the results.')
flags.DEFINE_integer('iterations', 100, 'number of iterations.')

# labels
import scipy.io as scio
labels = scio.loadmat('/home/huawei/rise/paper_2/dataset/dynamic_datasets/{}_labels.mat'.format(FLAGS.data_name))['labels'][0]
def one_hot(row):
    arr = np.zeros((len(labels), len(set(labels))))
    for i in range(len(row)):
        arr[i, row[i]-1] = 1
    return arr
labels = one_hot(labels)

# preprocess
adjs, features = load_data(FLAGS.data_name)
adj = adjs[0]
feature = features[0]

adj_orig = sparse_to_tuple(adj)
adj_norm = preprocess_graph(adj)
feature = sparse_to_tuple(feature)

features_nonzero = feature[1].shape[0]
num_node = np.array(adjs[0]).shape[1]
feature_dim = np.array(features[0]).shape[1]

pos_weight = float(num_node * num_node - adj[1].sum()) / adj[1].sum()
norm = num_node * num_node / float((num_node * num_node - adj[1].sum()) * 2)

print('num_node: ', num_node, ' feature_dim: ', feature_dim, ' pos_weight: ', pos_weight, ' norm: ', norm)

# placeholder
placeholders = {
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'adj_norm': tf.sparse_placeholder(tf.float32),
    'feature': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
}

# model
hidden1 = GraphConvolutionSparse(input_dim=feature_dim,
                                  output_dim=FLAGS.hidden1,
                                  adj=placeholders['adj_norm'],
                                  features_nonzero=features_nonzero,
                                  act=tf.nn.relu,
                                  dropout=placeholders['dropout'])(placeholders['feature'])

noise = gaussian_noise_layer(hidden1, 0.1)

embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
                                   output_dim=FLAGS.hidden2,
                                   adj=placeholders['adj_norm'],
                                   act=lambda x: x,
                                   dropout=placeholders['dropout'])(noise)

# reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2, act=lambda x: x)(embeddings)
# label = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False), [-1])
# loss
# cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=reconstructions, targets=label,
#                                                                       pos_weight=pos_weight))

pred = Dense(FLAGS.hidden2, labels.shape[1])(embeddings)
cost = norm * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels))

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
opt_op = optimizer.minimize(cost)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(FLAGS.iterations):
    _, reconstruct_loss, embedding = sess.run([opt_op, cost, embeddings], feed_dict={placeholders['adj_orig']: adj_orig,
                                                              placeholders['adj_norm']: adj_norm,
                                                              placeholders['feature']: feature})
    print(reconstruct_loss)
print('train done')
print(embedding)
np.savetxt('/home/huawei/rise/paper_2/dataset/embedding/{}.txt'.format(FLAGS.data_name), embedding)
