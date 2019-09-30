import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_list('hidden3', [128, 64, 50], 'Number of units in TCN.')
flags.DEFINE_integer('discriminator_out', 0, 'discriminator_out.')
flags.DEFINE_float('discriminator_learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('learning_rate', .5*0.001, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('features', 0, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('seed', 50, 'seed for fixing the results.')
flags.DEFINE_integer('iterations', 100, 'number of iterations.')
# flags.DEFINE_integer('seq_len', 6, 'time stamp for each train.')
flags.DEFINE_float('time_decay', 0.8, 'decay for each step.')
flags.DEFINE_float('alpha', 1, 'contribution of structural loss.')


def get_settings(dataname, train_length):
    iterations = FLAGS.iterations
    re = {'data_name': dataname, 'iterations': iterations, 'train_length': train_length}
    return re

