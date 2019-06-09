"""
Embedding

"""
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class SentenceEmbedding(tf.keras.Model):

    def __init__(self, config):

        super(SentenceEmbedding, self).__init__()
        self.config = config
        self.word_embedding = layers.Embedding(config.embed_size, config.embed_dim)
        self.encoder = eval(config.encoder_type)(config)

    def call(self, x):
        sentence = self.word_embedding(x)
        embedding = self.encoder(sentence)

        return embedding

    def encode(self, x):
        embedding = self.encoder(x)
        return embedding

class LSTMEncoder(tf.keras.Model):
    """
    Basic LSTM Encoder
    """

    def __init__(self, config):
        super(LSTMEncoder, self).__init__()
        self.config = config
        self.rnn = layers.LSTM(
                           units=config.hidden_dim,
                           dropout=config.dropout
                            )
        self.batch_norm = layers.BatchNormalization()

    def call(self, x):

        embedding = self.rnn(x)
        embedding = self.batch_norm(embedding)

        return embedding


class BiLSTMMaxPoolEncoder(tf.keras.Model):
    """
    Bidirectional LSTM with max pooling
    """
    def __init__(self, config):
        super(BiLSTMMaxPoolEncoder, self).__init__()
        self.config = config
        self.rnn = layers.LSTM(
                           units=config.hidden_dim,
                           dropout=config.dropout,
                            )
        self.bidirectional = layers.Bidirectional

        self.max_pool = layers.MaxPool1D(1)

    def call(self, x):

        embedding = self.rnn(x)
        embedding = self.bidirectional(embedding)
        # Max pooling
        emb = self.max_pool(embedding)
        emb = emb.squeeze(2)
        return emb


class HBMP(tf.keras.Model):
    """
    Hierarchical Bi-LSTM Max Pooling Encoder
    """
    def __init__(self, config):
        super(HBMP, self).__init__()
        self.config = config
        self.max_pool = layers.MaxPool1D(1)

        self.cells = config.cells

        self.hidden_dim = config.hidden_dim
        self.rnn1 = layers.LSTM(
	        units=config.hidden_dim,
	        dropout=config.dropout,
        )
        self.rnn2 = layers.LSTM(
	        units=config.hidden_dim,
	        dropout=config.dropout,
        )
        self.rnn3 = layers.LSTM(
	        units=config.hidden_dim,
	        dropout=config.dropout,
        )
        self.bidirectional = layers.Bidirectional

    def call(self, x):

        emb1 = self.rnn1(x)
        emb1 = self.bidirectional(emb1)
        emb1 = self.max_pool(emb1)

        emb2 = self.rnn2(x)
        emb2 = self.bidirectional(emb2)
        emb2 = self.max_pool(emb2)

        emb3 = self.rnn3(x)
        emb3 = self.bidirectional(emb3)
        emb3 = self.max_pool(emb3)

        emb = tf.concat([emb1,emb2,emb3], axis=2)
        emb = emb.squeeze(0)

        return emb