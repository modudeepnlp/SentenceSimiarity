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
        self._outDense = tf.keras.layers.Dense(outdim, activation='softmax')

    def call(self, x):
        x =self._dropout(x)
        x = self._dense(x)
        x =self._dropout(x)
        x = self._dense(x)
        x = self._dense(x)

        return x


class SentenceEmbedding(tf.keras.Model):
    def __init__(self,vocab_len):
        super(SentenceEmbedding, self).__init__()

        units = FLAGS.dim
        dropout = FLAGS.dropout
        self._word_embedding = tf.keras.layers.Embedding(vocab_len, FLAGS.dim)
        
        self._lstm = tf.keras.layers.LSTM(units=units, dropout=dropout, return_sequences=True)
        self._pool = tf.keras.layers.MaxPool1D(1)
        self._bidirectional = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=units, dropout=dropout))

    def __call__(self, x):
        x = self._word_embedding(x)
        embedding = self._bidirectional(x)
        embedding = tf.reshape(embedding, [embedding.shape[0], embedding.shape[1], 1])
        embedding = self._pool(embedding)

        embedding1 = self._bidirectional(x)
        embedding1 = tf.reshape(embedding1, [embedding1.shape[0], embedding1.shape[1], 1])
        embedding1 = self._pool(embedding1)

        embedding2 = self._bidirectional(x)
        embedding2 = tf.reshape(embedding2, [embedding2.shape[0], embedding2.shape[1], 1])
        embedding2 = self._pool(embedding2)

        emb = tf.concat([embedding, embedding1, embedding2], axis=2) # asix ???
        #emb = emb.squeeze(0) #??
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
