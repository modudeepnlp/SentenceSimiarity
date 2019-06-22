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
        self._dense1 = tf.keras.layers.Dense(dim, activation='relu')
        self._outDense = tf.keras.layers.Dense(outdim)

    def call(self, x):
        x = self._dropout(x)
        x = self._dense(x)
        x = self._dropout(x)
        x = self._dense1(x)
        x = self._outDense(x)

        return x


class SentenceEmbedding(tf.keras.Model):
    def __init__(self, vocab_len):
        super(SentenceEmbedding, self).__init__()

        units = FLAGS.dim
        dropout = FLAGS.dropout
        self._word_embedding = tf.keras.layers.Embedding(vocab_len, FLAGS.dim, input_length=FLAGS.length)

        self._pool = tf.keras.layers.GlobalMaxPool1D()
        self._bidirectional = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=units, dropout=dropout,return_sequences=True))
        self._bidirectional1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=units, dropout=dropout, return_sequences=True))
        self._bidirectional2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=units, dropout=dropout, return_sequences=True))

    def __call__(self, x):
        x = self._word_embedding(x)
        embedding = self._bidirectional(x)

        embedding = self._pool(embedding)

        embedding1 = self._bidirectional1(x)
        embedding1 = self._pool(embedding1)

        embedding2 = self._bidirectional2(x)
        embedding2 = self._pool(embedding2)

        emb = tf.concat([embedding, embedding1, embedding2], axis=1)
        return emb


class HBMP(tf.keras.Model):
    def __init__(self, vocab_len):
        super(HBMP, self).__init__()

        self._embedding = SentenceEmbedding(vocab_len)
        self._fcl = FCL()

    def call(self, premise, hypothesis):
        x = self._embedding(premise)
        y = self._embedding(hypothesis)
        features = tf.concat([x, y, tf.abs(x - y), x * y], 1)
        output = self._fcl(features)
        return output
