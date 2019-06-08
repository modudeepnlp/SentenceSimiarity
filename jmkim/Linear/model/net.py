import tensorflow as tf


class Linear(tf.keras.Model):
    def __init__(self, vocab):
        super(Linear, self).__init__()



        # self.embeedding = tf.nn.embedding_lookup(vocab_len, 32)
        # self.layer = tf.contrib.layers.fully_connected(32, 32)
        # self.output = tf.contrib.layers.fully_connected(32, 4)

    def call(self, x, y):
        self._x = self.embeedding(x)
        sentence1 = self.layer(self._x)
        sentence1 = sentence1.mean(1)

        self._y = self.embeedding(y)
        sentence2 = self.layer(self._y)
        sentence2 = sentence2.mean(1)

        self.result = self.output(sentence2 - sentence1)
        return self.result
