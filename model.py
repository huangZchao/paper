from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, TCN, Dense
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
        self.struct_features = placeholders['struct_features']
        self.struct_adj_norms = placeholders['struct_adj_norms']
        self.features_nonzeros = features_nonzeros
        self.feature_dim = feature_dim

        self.dropout = placeholders['dropout']
        self.num_node = num_node
        self.seq_len = seq_len
        self.num_channel = num_channel
        self.build()

    def _build(self):
        with tf.variable_scope('Encoder', reuse=None):
            self.embeddings = []
            self.reconstructions = []
            for ts, (struct_adj_norm, struct_feature) in enumerate(zip(self.struct_adj_norms, self.struct_features)):
                features_nonzero = self.features_nonzeros[ts]
                self.hidden1 = GraphConvolutionSparse(input_dim=self.feature_dim,
                                                      output_dim=FLAGS.hidden1,
                                                      adj=struct_adj_norm,
                                                      features_nonzero=features_nonzero,
                                                      act=tf.nn.relu,
                                                      dropout=self.dropout,
                                                      logging=self.logging,
                                                      name='e_dense_1_{}'.format(ts))(struct_feature)

                self.noise = gaussian_noise_layer(self.hidden1, 0.1)

                embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
                                                   output_dim=FLAGS.hidden2,
                                                   adj=struct_adj_norm,
                                                   act=lambda x: x,
                                                   dropout=self.dropout,
                                                   logging=self.logging,
                                                   name='e_dense_2_{}'.format(ts))(self.noise)

                # for auxilary loss
                reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                      logging=self.logging)(embeddings)

                self.embeddings.append(tf.reshape(embeddings, [self.num_node, 1, FLAGS.hidden2]))
                self.reconstructions.append(reconstructions)

            # TCN
            sequence = tf.concat(self.embeddings, axis=1, name='concat_embedding')
            self.sequence_out = TCN(num_channels=[FLAGS.hidden3], sequence_length=self.seq_len)(sequence)
            self.reconstructions_tss = []

            # Dense
            for ts in range(self.seq_len):
                reconstructions_ts = Dense(input_dim=FLAGS.hidden3, classes=self.num_node)(self.sequence_out[:, ts, :])
                reconstructions_ts = tf.reshape(reconstructions_ts, [-1])
                self.reconstructions_tss.append(reconstructions_ts)


def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise
