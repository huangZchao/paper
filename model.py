from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, Attention
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


class ARGA(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(ARGA, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adjs = placeholders['adjs']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        emb_list = []
        with tf.variable_scope('Encoder', reuse=None):
            for hop_num, adj in enumerate(self.adjs):
                self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                                      output_dim=FLAGS.hidden1,
                                                      adj=adj,
                                                      features_nonzero=self.features_nonzero,
                                                      act=tf.nn.relu,
                                                      dropout=self.dropout,
                                                      logging=self.logging,
                                                      name='e_dense_1_{}'.format(hop_num))(self.inputs)

                self.noise = gaussian_noise_layer(self.hidden1, 0.1)

                embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
                                                   output_dim=FLAGS.hidden2,
                                                   adj=adj,
                                                   act=lambda x: x,
                                                   dropout=self.dropout,
                                                   logging=self.logging,
                                                   name='e_dense_2_{}'.format(hop_num))(self.noise)
                emb_list.append(embeddings)

            # simple attention
            # todo bad attention weight cal
            atts = []
            for hop_num, embed in enumerate(emb_list):
                # 'alpha' shape: [n_vertices, 1]
                alpha = Attention(input_dim=FLAGS.hidden2,
                                  output_dim=1,
                                  dropout=self.dropout,
                                  logging=self.logging,
                                  name='e_attent_{}'.format(hop_num))(embed)
                atts.append(alpha)
            # cal attention weight
            att_sum = 0
            for att in atts:
                att_sum += tf.exp(att)
            alphas = []
            for att in atts:
                alphas.append(tf.exp(att)/att_sum)
            self.embeddings = 0
            for alpha, embed in zip(alphas, emb_list):
                self.embeddings += tf.multiply(alpha, embed)

            self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                       act=lambda x: x,
                                                       logging=self.logging)(self.embeddings)


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


class Discriminator(Model):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)

        self.act = tf.nn.relu

    def construct(self, inputs, reuse = False):
        # with tf.name_scope('Discriminator'):
        with tf.variable_scope('Discriminator'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            # np.random.seed(1)
            tf.set_random_seed(1)
            dc_den1 = tf.nn.relu(dense(inputs, FLAGS.hidden2, FLAGS.hidden3, name='dc_den1'))
            dc_den2 = tf.nn.relu(dense(dc_den1, FLAGS.hidden3, FLAGS.hidden1, name='dc_den2'))
            output = dense(dc_den2, FLAGS.hidden1, 1, name='dc_output')
            return output

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise
