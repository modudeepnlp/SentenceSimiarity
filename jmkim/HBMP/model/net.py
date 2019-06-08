import tensorflow as tf
from tensorflow.python.keras import backend as K
from config import FLAGS


class FCL(tf.keras.Model):
    def __init__(self, **kwargs):
        super(FCL, self).__init__(**kwargs)

        dropout = FLAGS.dropout
        dim = FLAGS.dim
        outdim = FLAGS.out_dim

        self._dropout = tf.keras.layers.Dropout(dropout)
        self._dense = tf.keras.layers.Dense(dim, activation='relu')
        self._outDense = tf.keras.layers.Dense(outdim)

    def call(self, x):
        x = tf.keras.layers.Dropout(x)
        x = tf.keras.layers.Dense(x)
        x = tf.keras.layers.Dropout(x)
        x = tf.keras.layers.Dense(x)
        x = tf.keras.layers.Dense(x)

        return x


class SentenceEmbedding():
    def __init__(self):
        super(SentenceEmbedding, self).__init__()

        units = FLAGS.length
        keep_prob = FLAGS.learning_rate
        self._lstm = tf.keras.layers.LSTMCell(units=units)
        self._dropout = tf.nn.RNNCellDropoutWrapper(self._lstm, output_keep_prob=keep_prob)

    def __call__(self, x):
        self._initShape = tf.zeros([FLAGS.batch_size, FLAGS.hidden_size], dtype=tf.float32)
        self._forward = self._lstm(x)
        self._backward = self._lstm(x)
        X = tf.unstack(x, 4, 1)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(self._forward, self._backward, X, dtype=tf.float32)
        return outputs


class HBMP(tf.keras.Model):
    def __init__(self):
        super(HBMP, self).__init__()

        self._embedding = SentenceEmbedding()
        self._fcl = FCL()

    def call(self, premise, hypothesis):
        x = self._embedding(premise)
        y = self._embedding(hypothesis)
        features = tf.concat([x, y, tf.abs(x - y), x * y], 1)
        output = self._fcl(features)
        return output
