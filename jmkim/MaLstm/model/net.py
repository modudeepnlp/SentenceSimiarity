import tensorflow as tf
from tensorflow.python.keras import backend as K


class MaLSTM(tf.keras.Model):
    def __init__(self, len, dim, vocab_size):
        super(MaLSTM, self).__init__()
        self._len = len
        self._dim = dim
        self._vocab_size = vocab_size

        self._embedding = tf.keras.layers.Embedding(self._vocab_size, self._dim, input_length=self._len)
        self._lstm = tf.keras.layers.LSTM(70)

        self._malstm_dist = ManDist()


    def call(self, x, y):

        q1 = self._embedding(x)
        q2 = self._embedding(y)

        q1_lstm = self._lstm(q1)
        q2_lstm = self._lstm(q2)

        malstm_dist = self._malstm_dist([q1_lstm, q2_lstm])
        return malstm_dist

class ManDist(tf.keras.Model):
    def __init__(self, **kwargs):
        super(ManDist, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result