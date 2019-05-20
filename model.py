from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, TCN
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCN(Model):
    def __init__(self, placeholders, feature_dim, features_nonzeros, num_node, seq_len, num_channel, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = feature_dim
        self.features_nonzeros = features_nonzeros
        self.adj_norms = placeholders['adj_norms']
        self.dropout = placeholders['dropout']
        self.num_node = num_node
        self.seq_len = seq_len
        self.num_channel = num_channel
        self.build()

    def _build(self):
        with tf.variable_scope('Encoder', reuse=None):
            self.embeddings = []
            self.reconstructions = []
            for ts, (adj_norm, inputs) in enumerate(zip(self.adj_norms, self.inputs)):
                features_nonzero = self.features_nonzeros[ts]
                self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                                      output_dim=FLAGS.hidden1,
                                                      adj=adj_norm,
                                                      features_nonzero=features_nonzero,
                                                      act=tf.nn.relu,
                                                      dropout=self.dropout,
                                                      logging=self.logging,
                                                      name='e_dense_1_{}'.format(ts))(inputs)

                self.noise = gaussian_noise_layer(self.hidden1, 0.1)

                embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
                                                   output_dim=FLAGS.hidden2,
                                                   adj=adj_norm,
                                                   act=lambda x: x,
                                                   dropout=self.dropout,
                                                   logging=self.logging,
                                                   name='e_dense_2_{}'.format(ts))(self.noise)

                # for auxilary loss
                reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                           act=lambda x: x,
                                                           logging=self.logging)(embeddings)

                self.embeddings.append(tf.reshape(embeddings, [1, 1, -1]))
                self.reconstructions.append(reconstructions)

            # TCN
            sequence = tf.concat(self.embeddings, axis=1, name='concat_embedding')
            sequence_out = TCN(num_channels=[self.num_node*FLAGS.hidden3]*self.num_channel, sequence_length=self.seq_len)(sequence)
            self.reconstructions_tss = []
            for ts in range(self.seq_len):
                reconstructions_ts = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                         act=lambda x: x,
                                                         logging=self.logging)(sequence_out[0, ts, :, :])
                self.reconstructions_tss.append(reconstructions_ts)


def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.variable_scope(name, reuse=None):
        # np.random.seed(1)
        tf.set_random_seed(1)
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise
